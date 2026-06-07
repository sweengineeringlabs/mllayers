use mlautograd::{MlResult, Tensor};
use crate::api::layer::Layer;

/// Sequential container that chains layers in order.
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Layer for Sequential {
    fn forward(&mut self, input: &Tensor) -> MlResult<Tensor> {
        let mut x = input.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers
            .iter_mut()
            .flat_map(|l| l.parameters_mut())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::layers::linear::Linear;
    use crate::core::layers::activations::relu::ReLU;

    #[test]
    fn test_sequential_new_stores_layers() {
        let seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(ReLU::new()),
        ]);
        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn test_sequential_forward_chains_layers() {
        let mut seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(ReLU::new()),
        ]);
        let input = Tensor::randn([2, 4]);
        let output = seq.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_sequential_parameters_aggregates_all_layers() {
        let seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(Linear::new(3, 2)),
        ]);
        assert_eq!(seq.parameters().len(), 4);
    }
}
