use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{self, BufReader, BufWriter},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct HyperParam {
    pub name: String,
    pub dim: usize,        // Embedding
    pub ffn_dim: usize,    // Expanded dimension
    pub layer: usize,      // number of layers
    pub head: usize,       // number of query heads
    pub kv_head: usize,    // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    pub seq_len: usize,    // max sequence length
}

impl HyperParam {
    pub fn from_yaml(file: File) -> io::Result<Self> {
        let reader = BufReader::new(file);
        match serde_yaml::from_reader(reader) {
            Ok(hp) => Ok(hp),
            Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        }
    }

    #[allow(unused)]
    pub fn to_yaml(&self, file: File) -> io::Result<()> {
        let writer = BufWriter::new(file);
        match serde_yaml::to_writer(writer, &self) {
            Ok(_) => Ok(()),
            Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        }
    }
}
