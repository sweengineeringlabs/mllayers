use crate::api::traits::layer::Layer;

/// Sequential container that chains layers in order.
pub struct Sequential {
    pub(crate) layers: Vec<Box<dyn Layer>>,
}
