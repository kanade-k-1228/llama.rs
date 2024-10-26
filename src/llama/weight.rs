use std::{
    fs::File,
    io::{self, Read},
};

use super::param::HyperParam;

#[derive(Debug)]
pub struct Weight {
    pub tok_emb_table: Vec<Vec<f32>>, // [vocab_size, dim]
    pub attn_norm: Vec<Vec<f32>>,     // [layer, dim]
    pub attn_wq: Vec<Vec<Vec<f32>>>,  // [layer, dim, dim]
    pub attn_wk: Vec<Vec<Vec<f32>>>,  // [layer, dim, dim]
    pub attn_wv: Vec<Vec<Vec<f32>>>,  // [layer, dim, dim]
    pub attn_wo: Vec<Vec<Vec<f32>>>,  // [layer, dim, dim]
    pub ffn_norm: Vec<Vec<f32>>,      // [layer, dim]
    pub ffn_w1: Vec<Vec<Vec<f32>>>,   // [layer, ffn_dim, dim]
    pub ffn_w2: Vec<Vec<Vec<f32>>>,   // [layer, dim, ffn_dim]
    pub ffn_w3: Vec<Vec<Vec<f32>>>,   // [layer, ffn_dim, dim]
    pub final_norm: Vec<f32>,         // [dim]
    pub rope_cos: Vec<Vec<f32>>,      // [seq_len, (dim/head)/2]
    pub rope_sin: Vec<Vec<f32>>,      // [seq_len, (dim/head)/2]
}

impl Weight {
    pub fn from_file(hp: &HyperParam, mut file: File) -> io::Result<Self> {
        Ok(Self {
            tok_emb_table: Self::read_tensor2d(&mut file, (hp.vocab_size, hp.dim))?,
            attn_norm: Self::read_tensor2d(&mut file, (hp.layer, hp.dim))?,
            attn_wq: Self::read_tensor3d(&mut file, (hp.layer, hp.dim, hp.dim))?,
            attn_wk: Self::read_tensor3d(&mut file, (hp.layer, hp.dim, hp.dim))?,
            attn_wv: Self::read_tensor3d(&mut file, (hp.layer, hp.dim, hp.dim))?,
            attn_wo: Self::read_tensor3d(&mut file, (hp.layer, hp.dim, hp.dim))?,
            ffn_norm: Self::read_tensor2d(&mut file, (hp.layer, hp.dim))?,
            ffn_w1: Self::read_tensor3d(&mut file, (hp.layer, hp.ffn_dim, hp.dim))?,
            ffn_w2: Self::read_tensor3d(&mut file, (hp.layer, hp.dim, hp.ffn_dim))?,
            ffn_w3: Self::read_tensor3d(&mut file, (hp.layer, hp.ffn_dim, hp.dim))?,
            final_norm: Self::read_tensor1d(&mut file, hp.dim)?,
            rope_cos: Self::read_tensor2d(&mut file, (hp.seq_len, hp.dim / hp.head / 2))?,
            rope_sin: Self::read_tensor2d(&mut file, (hp.seq_len, hp.dim / hp.head / 2))?,
        })
    }

    fn read_tensor1d(file: &mut File, dim: usize) -> io::Result<Vec<f32>> {
        let mut tensor = vec![0.0; dim];
        file.read_exact(bytemuck::cast_slice_mut(&mut tensor))?;
        Ok(tensor)
    }

    fn read_tensor2d(file: &mut File, dim: (usize, usize)) -> io::Result<Vec<Vec<f32>>> {
        let (rows, cols) = dim;
        let mut tensor = Vec::with_capacity(rows);
        for _ in 0..rows {
            let row = Self::read_tensor1d(file, cols)?;
            tensor.push(row);
        }
        Ok(tensor)
    }

    fn read_tensor3d(
        file: &mut File,
        dim: (usize, usize, usize),
    ) -> io::Result<Vec<Vec<Vec<f32>>>> {
        let (depth, rows, cols) = dim;
        let mut tensor = Vec::with_capacity(depth);
        for _ in 0..depth {
            let matrix = Self::read_tensor2d(file, (rows, cols))?;
            tensor.push(matrix);
        }
        Ok(tensor)
    }
}
