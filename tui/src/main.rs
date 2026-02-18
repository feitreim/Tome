use std::io;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use tokio::sync::mpsc;

const SERVER_URL: &str = "http://0.0.0.0:8080";
const SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer concisely and clearly.";
const MAX_TOKENS: u32 = 32768;
const TEMPERATURE: f32 = 0.6;

#[derive(Clone)]
enum Message {
    User(String),
    Assistant(String),
}

struct App {
    messages: Vec<Message>,
    input: String,
    scroll: u16,
    streaming: bool,
}

impl App {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            scroll: 0,
            streaming: false,
        }
    }

    fn submit(&mut self) -> Option<String> {
        if self.input.trim().is_empty() || self.streaming {
            return None;
        }
        let user_text = self.input.drain(..).collect::<String>();
        self.messages.push(Message::User(user_text));
        self.messages.push(Message::Assistant(String::new()));
        self.streaming = true;
        self.scroll = u16::MAX;
        Some(self.build_chatml_prompt())
    }

    /// Format full conversation as a ChatML prompt.
    /// Uses non-thinking mode: history excludes thinking content,
    /// and we prime the assistant turn with <think>\n\n</think>\n to suppress thinking.
    fn build_chatml_prompt(&self) -> String {
        let mut prompt = format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n");
        for msg in &self.messages {
            match msg {
                Message::User(text) => {
                    prompt.push_str(&format!("<|im_start|>user\n{text}<|im_end|>\n"));
                }
                Message::Assistant(text) if !text.is_empty() => {
                    // History turns: only include final output (no thinking content)
                    prompt.push_str(&format!("<|im_start|>assistant\n{text}<|im_end|>\n"));
                }
                Message::Assistant(_) => {
                    // Current (empty) turn: prime for non-thinking mode
                    prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n");
                }
            }
        }
        prompt
    }

    fn append_raw(&mut self, text: &str) {
        if let Some(Message::Assistant(ref mut t)) = self.messages.last_mut() {
            t.push_str(text);
        }
        self.scroll = u16::MAX;
    }

    fn finish_stream(&mut self) {
        self.streaming = false;
    }
}

enum StreamEvent {
    Text(String),
    Done,
    Error(String),
}

fn spawn_sse_request(prompt: String, tx: mpsc::UnboundedSender<StreamEvent>) {
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let res = client
            .post(format!("{SERVER_URL}/v1/completions"))
            .json(&serde_json::json!({
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            }))
            .send()
            .await;

        let resp: reqwest::Response = match res {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(format!("request failed: {e}")));
                let _ = tx.send(StreamEvent::Done);
                return;
            }
        };

        let mut stream = resp.bytes_stream();
        let mut buf = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(format!("stream error: {e}")));
                    break;
                }
            };

            buf.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim().to_string();
                buf = buf[pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                let Some(data) = line
                    .strip_prefix("data: ")
                    .or_else(|| line.strip_prefix("data:"))
                else {
                    continue;
                };

                if data.trim() == "[DONE]" {
                    let _ = tx.send(StreamEvent::Done);
                    return;
                }

                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data.trim()) {
                    if let Some(choices) = parsed["choices"].as_array() {
                        if let Some(choice) = choices.first() {
                            if let Some(text) = choice["text"].as_str() {
                                if !text.is_empty() {
                                    let _ = tx.send(StreamEvent::Text(text.to_string()));
                                }
                            }
                            if let Some(reason) = choice["finish_reason"].as_str() {
                                if reason == "stop" || reason == "length" {
                                    let _ = tx.send(StreamEvent::Done);
                                    return;
                                }
                            }
                        }
                    }
                    if let Some(err) = parsed["error"].as_str() {
                        let _ = tx.send(StreamEvent::Error(err.to_string()));
                        let _ = tx.send(StreamEvent::Done);
                        return;
                    }
                }
            }
        }

        let _ = tx.send(StreamEvent::Done);
    });
}

fn build_chat_lines(app: &App) -> Vec<Line<'_>> {
    let mut lines = Vec::new();
    for msg in &app.messages {
        match msg {
            Message::User(text) => {
                lines.push(Line::from(vec![
                    Span::styled(
                        "You: ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(text.as_str()),
                ]));
            }
            Message::Assistant(text) => {
                lines.push(Line::from(vec![
                    Span::styled(
                        "Model: ",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(text.as_str()),
                ]));
            }
        }
        lines.push(Line::raw(""));
    }
    lines
}

fn draw(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> io::Result<()> {
    terminal.draw(|frame| {
        let area = frame.area();
        let chunks = Layout::vertical([Constraint::Min(3), Constraint::Length(3)]).split(area);

        // Clamp scroll to content height
        let inner_width = chunks[0].width.saturating_sub(2) as usize;
        let content_height: u16 = build_chat_lines(app)
            .iter()
            .map(|line| {
                let text_len: usize = line.spans.iter().map(|s| s.content.len()).sum();
                if inner_width == 0 {
                    1
                } else {
                    ((text_len / inner_width) + 1) as u16
                }
            })
            .sum();
        let visible_height = chunks[0].height.saturating_sub(2);
        let max_scroll = content_height.saturating_sub(visible_height);
        if app.scroll > max_scroll {
            app.scroll = max_scroll;
        }

        let chat = Paragraph::new(build_chat_lines(app))
            .block(Block::default().borders(Borders::ALL).title(" Tome Chat "))
            .wrap(Wrap { trim: false })
            .scroll((app.scroll, 0));

        frame.render_widget(chat, chunks[0]);

        // Input box
        let status = if app.streaming {
            " (streaming...) "
        } else {
            ""
        };
        let input = Paragraph::new(app.input.as_str()).block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Input{status} ")),
        );
        frame.render_widget(input, chunks[1]);

        frame.set_cursor_position((chunks[1].x + 1 + app.input.len() as u16, chunks[1].y + 1));
    })?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let mut app = App::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();

    loop {
        draw(&mut terminal, &mut app)?;

        if event::poll(std::time::Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Esc => break,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Enter => {
                        if let Some(prompt) = app.submit() {
                            spawn_sse_request(prompt, tx.clone());
                        }
                    }
                    KeyCode::Backspace => {
                        app.input.pop();
                    }
                    KeyCode::Char(c) => {
                        app.input.push(c);
                    }
                    KeyCode::Up => {
                        app.scroll = app.scroll.saturating_sub(1);
                    }
                    KeyCode::Down => {
                        app.scroll = app.scroll.saturating_add(1);
                    }
                    _ => {}
                }
            }
        }

        while let Ok(ev) = rx.try_recv() {
            match ev {
                StreamEvent::Text(text) => app.append_raw(&text),
                StreamEvent::Done => app.finish_stream(),
                StreamEvent::Error(e) => {
                    app.append_raw(&format!("\n[error: {e}]"));
                    app.finish_stream();
                }
            }
        }
    }

    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}
