use std::collections::{HashMap, HashSet};

pub type NodeId = String;

#[derive(Default)]
struct TrieNode {
    children: HashMap<u32, TrieNode>,
    cached_nodes: HashSet<NodeId>,
}

/// Radix trie tracking which inference nodes have cached which token prefixes.
#[derive(Default)]
pub struct CacheTrie {
    root: TrieNode,
}

impl CacheTrie {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark that `node_id` has the given token sequence cached.
    pub fn insert(&mut self, node_id: &str, tokens: &[u32]) {
        let mut current = &mut self.root;
        for &token in tokens {
            current = current.children.entry(token).or_default();
            current.cached_nodes.insert(node_id.to_string());
        }
    }

    /// Find nodes with the longest matching prefix, returning (node_id, match_length) pairs.
    pub fn find_best_match(&self, tokens: &[u32]) -> Vec<(NodeId, usize)> {
        let mut current = &self.root;
        let mut best: HashMap<NodeId, usize> = HashMap::new();

        for (i, &token) in tokens.iter().enumerate() {
            match current.children.get(&token) {
                Some(child) => {
                    for node_id in &child.cached_nodes {
                        best.insert(node_id.clone(), i + 1);
                    }
                    current = child;
                }
                None => break,
            }
        }

        let max_len = best.values().copied().max().unwrap_or(0);
        best.into_iter()
            .filter(|(_, len)| *len == max_len)
            .collect()
    }

    /// Remove a cached prefix for a node.
    pub fn evict(&mut self, node_id: &str, tokens: &[u32]) {
        Self::evict_recursive(&mut self.root, node_id, tokens, 0);
    }

    fn evict_recursive(node: &mut TrieNode, node_id: &str, tokens: &[u32], depth: usize) {
        if depth < tokens.len() {
            if let Some(child) = node.children.get_mut(&tokens[depth]) {
                child.cached_nodes.remove(node_id);
                Self::evict_recursive(child, node_id, tokens, depth + 1);
                if child.children.is_empty() && child.cached_nodes.is_empty() {
                    node.children.remove(&tokens[depth]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_find() {
        let mut trie = CacheTrie::new();
        trie.insert("node-1", &[1, 2, 3, 4]);
        trie.insert("node-2", &[1, 2, 5, 6]);

        let matches = trie.find_best_match(&[1, 2, 3, 4, 5]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, "node-1");
        assert_eq!(matches[0].1, 4);
    }

    #[test]
    fn no_match() {
        let trie = CacheTrie::new();
        let matches = trie.find_best_match(&[1, 2, 3]);
        assert!(matches.is_empty());
    }

    #[test]
    fn evict_prefix() {
        let mut trie = CacheTrie::new();
        trie.insert("node-1", &[1, 2, 3]);
        trie.evict("node-1", &[1, 2, 3]);
        let matches = trie.find_best_match(&[1, 2, 3]);
        assert!(matches.is_empty());
    }
}
