use mlautograd::{MlResult, Tensor};
use crate::api::traits::layer::Layer;
use crate::api::types::sequential::Sequential;

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
        self.layers.iter_mut().flat_map(|l| l.parameters_mut()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::linear::Linear;
    use crate::api::types::relu::Relu;

    // @covers: forward
    #[test]
    fn test_forward_chains_layers_in_order() {
        let mut seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(Relu::new()),
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

    // @covers: parameters_mut
    #[test]
    fn test_parameters_mut_aggregates_all_sublayer_params() {
        let mut seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
        ]);
        assert_eq!(seq.parameters_mut().len(), 2);
    }
}
