use crate::api::traits::layer::Layer;

/// Sequential container that chains layers in order.
pub struct Sequential {
    pub(crate) layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self { Self { layers } }
    pub fn len(&self) -> usize { self.layers.len() }
    pub fn is_empty(&self) -> bool { self.layers.is_empty() }
}
