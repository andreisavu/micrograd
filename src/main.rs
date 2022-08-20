use std::iter::zip;
use std::ops;
use std::{cell::RefCell, rc::Rc};

// Implementation based on the following tutorial by Andrej Karpathy going into
// the details on how he built the micrograd library in Python. Doing the same
// in Rust is not entirely trivial given how the compiler handles state.

// See the video at: https://www.youtube.com/watch?v=VMj-3S1tku0
// https://github.com/karpathy/micrograd

//
// Step 1. Define the core state object and the mutable expression graph.
//

#[derive(Clone, Debug)]
enum Op {
    None,
    Plus,
    Multiply,
    Tanh,
    Exp,
    Pow
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

    fn child(&self, index: usize) -> Value {
        self.0.borrow().children[index].clone()
    }

    fn only_child(&self) -> Value {
        assert!(self.0.borrow().children.len() == 1);
        self.child(0)
    }

    fn lhs(&self) -> Value {
        self.child(0)
    }

    fn rhs(&self) -> Value {
        self.child(1)
    }

    fn inc_gradient(&self, amount: f64) {
        self.0.borrow_mut().gradient += amount
    }

    fn gradient(&self) -> f64 {
        self.0.borrow().gradient
    }

    fn value(&self) -> f64 {
        self.0.borrow().value
    }

    fn op(&self) -> Op {
        self.0.borrow().op.clone()
    }
}

//
// Step 2. Define Standard mathematical operations on Value objects.
//

impl Value {
    fn tanh(self) -> Value {
        Value(Rc::new(RefCell::new(ValueState {
            value: self.value().tanh(),
            children: vec![self.clone()],
            gradient: 0.0,
            op: Op::Tanh,
        })))
    }

    fn exp(self) -> Value {
        Value(Rc::new(RefCell::new(ValueState {
            value: self.value().exp(),
            children: vec![self.clone()],
            gradient: 0.0,
            op: Op::Exp,
        })))
    }

    fn pow(self, value: f64) -> Value {
        Value(Rc::new(RefCell::new(ValueState {
            value: self.value().powf(value),
            children: vec![self.clone()],
            gradient: 0.0,
            op: Op::Pow,
        })))
    }
}

impl ops::Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Value(Rc::new(RefCell::new(ValueState {
            value: self.value() + rhs.value(),
            children: vec![self.clone(), rhs.clone()],
            gradient: 0.0,
            op: Op::Plus,
        })))
    }
}

impl ops::Add<f64> for Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Value {
        self.0.borrow_mut().value += rhs;
        self
    }
}

impl ops::Sub<f64> for Value {
    type Output = Value;

    fn sub(self, rhs: f64) -> Self::Output {
        self.0.borrow_mut().value -= rhs;
        self
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        Value(Rc::new(RefCell::new(ValueState {
            value: self.value() * rhs.value(),
            children: vec![self.clone(), rhs.clone()],
            gradient: 0.0,
            op: Op::Multiply,
        })))
    }
}

//
// Step 3. Each Value object can know its local gradient with respect to
// the overall expression. This function calculates that gradient value.
//

fn compute_gradients(out: &Value) {
    out.0.borrow_mut().gradient = 1.0;

    fn _reset_children_gradients(node: &Value) {
        for children in node.0.borrow().children.iter() {
            children.0.borrow_mut().gradient = 0.0;
            _reset_children_gradients(children);
        }
    }

    fn _compute_children_gradients(node: &Value) {
        let out_gradient = node.gradient();
        let out_value = node.value();

        match node.op() {
            Op::Plus => {
                node.lhs().inc_gradient(out_gradient);
                node.rhs().inc_gradient(out_gradient);
            }
            Op::Multiply => {
                node.lhs().inc_gradient(node.rhs().value() * out_gradient);
                node.rhs().inc_gradient(node.lhs().value() * out_gradient);
            }
            Op::Tanh => {
                node.only_child()
                    .inc_gradient((1.0 - out_value * out_value) * out_gradient);
            }
            Op::Exp => {
                node.only_child().inc_gradient(out_value * out_gradient);
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

//
// Step 4. Assemble Values into a Neuron structure that use the tanh activation function.
//

struct Neuron {
    output: Value,
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
            output: (output + bias_value).tanh(),
        }
    }
}

fn main() {
    let neuron = Neuron::new(&[2.0, 0.0], &[-3.0, 1.0], 6.8813735870195432);

    compute_gradients(&neuron.output);

    dump(&neuron.output);
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
