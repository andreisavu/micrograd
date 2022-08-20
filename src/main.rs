use std::ops;
use std::{cell::RefCell, rc::Rc};
use std::iter::zip;

#[derive(Debug)]
enum Op {
    None,
    Plus,
    Multiply,
    Tanh,
}

#[derive(Debug)]
struct ValueState {
    value: f64,
    children: Vec<Value>,
    gradient: f64,
    op: Op,
}

#[derive(Clone, Debug)]
struct Value(Rc<RefCell<ValueState>>);

impl Value {
    fn new(value: f64) -> Value {
        Value(Rc::new(RefCell::new(ValueState {
            value: value,
            children: Vec::new(),
            gradient: 0.0,
            op: Op::None,
        })))
    }

    fn tanh(self) -> Value {
        Value(Rc::new(RefCell::new(ValueState {
            value: self.0.borrow().value.tanh(),
            children: vec![self.clone()],
            gradient: 0.0,
            op: Op::Tanh,
        })))
    }
}

impl ops::Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Value(Rc::new(RefCell::new(ValueState {
            value: self.0.borrow().value + rhs.0.borrow().value,
            children: vec![self.clone(), rhs.clone()],
            gradient: 0.0,
            op: Op::Plus,
        })))
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        Value(Rc::new(RefCell::new(ValueState {
            value: self.0.borrow().value * rhs.0.borrow().value,
            children: vec![self.clone(), rhs.clone()],
            gradient: 0.0,
            op: Op::Multiply,
        })))
    }
}

fn compute_gradients(out: &Value) {
    out.0.borrow_mut().gradient = 1.0;

    fn _reset_children_gradients(node: &Value) {
        for children in node.0.borrow().children.iter() {
            children.0.borrow_mut().gradient = 0.0;
            _reset_children_gradients(children);
        }
    }

    fn _compute_children_gradients(node: &Value) {
        let out_gradient = node.0.borrow().gradient;
        let out_value = node.0.borrow().value;
        match node.0.borrow().op {
            Op::Plus => {
                node.0.borrow().children[0].0.borrow_mut().gradient += out_gradient;
                node.0.borrow().children[1].0.borrow_mut().gradient += out_gradient;
            }
            Op::Multiply => {
                node.0.borrow().children[0].0.borrow_mut().gradient +=
                    node.0.borrow().children[1].0.borrow().value * out_gradient;
                node.0.borrow().children[1].0.borrow_mut().gradient +=
                    node.0.borrow().children[0].0.borrow_mut().value * out_gradient;
            }
            Op::Tanh => {
                node.0.borrow().children[0].0.borrow_mut().gradient +=
                    (1.0 - out_value * out_value) * out_gradient;
            }
            _ => (),
        }
        for children in node.0.borrow().children.iter() {
            _compute_children_gradients(children);
        }
    }

    _reset_children_gradients(out);
    _compute_children_gradients(out)
}

fn dump(out: &Value) {
    fn _dump(out: &Value, prefix: String) {
        println!(
            "{}V={:?} O={:?} G={:?} ID={:?}",
            prefix,
            out.0.borrow().value,
            out.0.borrow().op,
            out.0.borrow().gradient,
            out.0.as_ptr()
        );
        for children in out.0.borrow().children.iter() {
            _dump(children, format!("{}   ", prefix))
        }
    }
    _dump(out, format!(""));
}

struct Neuron {
    output: Value
}

impl Neuron {
    fn new(inputs: &[f64], weights: &[f64], bias: f64) -> Neuron {
        let mut input_values = Vec::new();
        let mut weight_values = Vec::new();

        let mut output = Value::new(0.0);
        for (raw_input, raw_weight) in zip(inputs, weights) {
            let input = Value::new(*raw_input);
            input_values.push(input.clone());

            let weight = Value::new(*raw_weight);
            weight_values.push(weight.clone());

            output = output + input * weight;
        }

        let bias_value = Value::new(bias);
        Neuron {
            output: (output + bias_value).tanh()
        }
    }
}

fn main() {
    let neuron = Neuron::new(
        &[2.0, 0.0],
        &[-3.0, 1.0],
        6.8813735870195432
    );

    compute_gradients(&neuron.output);

    dump(&neuron.output);
}

// fn main() {
//     let a = Value::new(-2.0);
//     let b = Value::new(3.0);

//     let d = a.clone() + b.clone();
//     let e = a.clone() + b.clone();

//     let f = d * e;

//     compute_gradients(&f);

//     dump(&f);
// }

// fn main() {
//     let a = Value::new(3.0);
//     let b = a.clone() + a.clone();

//     compute_gradients(&b);

//     dump(&b);
// }
