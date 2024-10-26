use super::param::HyperParam;

pub struct Context {
    pub k_cache: Vec<Vec<Vec<f32>>>, // [layer, seq_len, dim]
    pub v_cache: Vec<Vec<Vec<f32>>>, // [layer, seq_len, dim]
}

impl Context {
    pub fn new(hp: &HyperParam) -> Self {
        Context {
            k_cache: vec![vec![vec![0.0; hp.dim]; hp.seq_len]; hp.layer],
            v_cache: vec![vec![vec![0.0; hp.dim]; hp.seq_len]; hp.layer],
        }
    }
}
