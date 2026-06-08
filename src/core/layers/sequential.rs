use mlautograd::{MlResult, Tensor};
use crate::api::traits::layer::Layer;
use crate::api::types::sequential::Sequential;

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
    use crate::api::types::linear::Linear;
    use crate::api::types::activations::ReLU;

    // @covers: new
    #[test]
    fn test_new_stores_layers() {
        let seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(ReLU::new()),
        ]);
        assert_eq!(seq.len(), 2);
    }

    // @covers: forward
    #[test]
    fn test_forward_chains_layers_in_order() {
        let mut seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(ReLU::new()),
        ]);
        let input = Tensor::randn([2, 4]);
        let output = seq.forward(&input).expect("forward");
        assert_eq!(output.shape(), &[2, 3]);
    }

    // @covers: parameters
    #[test]
    fn test_parameters_aggregates_all_sublayer_params() {
        let seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(Linear::new(3, 2)),
        ]);
        assert_eq!(seq.parameters().len(), 4);
    }

    // @covers: is_empty
    #[test]
    fn test_is_empty_returns_true_for_empty_container() {
        let seq = Sequential::new(vec![]);
        assert!(seq.is_empty());
    }
}
