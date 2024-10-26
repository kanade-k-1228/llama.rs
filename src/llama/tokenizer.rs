use super::param::*;
use std::{
    fs::File,
    io::{self, Cursor, Read},
};

pub struct Tokenizer {
    vocab: Vec<String>,
}

pub const SOS: usize = 1; // Start of Sequence
pub const EOS: usize = 2; // End of Sequence

impl Tokenizer {
    pub fn from_file(hp: &HyperParam, mut file: File) -> io::Result<Self> {
        let mut buf = Vec::<u8>::with_capacity(hp.vocab_size);
        file.read_to_end(&mut buf)?;
        Ok(Self::from_buf(&buf, hp))
    }
    fn from_buf(buf: &[u8], hp: &HyperParam) -> Self {
        let mut reader = Cursor::new(buf);
        let vocab = (0..hp.vocab_size)
            .map(|_| {
                let mut intbuf: [u8; 4] = [0; 4];
                reader.read_exact(&mut intbuf).unwrap();
                let strsize = u32::from_le_bytes(intbuf);
                let mut strbuf: Vec<u8> = vec![0; strsize as usize];
                reader.read_exact(&mut strbuf).unwrap();
                let vocab: String = strbuf.into_iter().map(|e| e as char).collect();
                vocab
            })
            .collect::<Vec<_>>();
        Self { vocab }
    }

    pub fn tok_to_str(&self, tok: usize) -> &String {
        self.vocab.get(tok).unwrap()
    }

    pub fn str_to_tok(&self, mut text: String) -> Vec<usize> {
        let mut ret: Vec<usize> = vec![];
        while text.len() > 0 {
            let hits = self.vocab.iter().enumerate().filter_map(|(idx, voc)| {
                if text.starts_with(voc) {
                    Some((idx, voc, voc.len()))
                } else {
                    None
                }
            });
            let max = hits.max_by(|(_, _, lhs), (_, _, rhs)| lhs.cmp(&rhs));
            if let Some((idx, _, len)) = max {
                ret.push(idx);
                text = text[len..].to_string();
            } else {
                // Tokenize as ASCII
                println!("CANNOT FIND IN VOCAB!:{:?}", text.chars().next().unwrap());
                let first_char = text.remove(0);
                for byte in first_char.to_string().as_bytes() {
                    ret.push(*byte as usize);
                }
            }
        }
        ret
    }
}
