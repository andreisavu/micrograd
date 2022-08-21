use rand::distributions::{Distribution, Uniform};
use std::iter::zip;
use std::ops;
use std::{cell::RefCell, rc::Rc};

// Implementation based on the following tutorial by Andrej Karpathy going into
// the details on how he built the micrograd library in Python.

// See the video at: https://www.youtube.com/watch?v=VMj-3S1tku0
// https://github.com/karpathy/micrograd

// Doing the same in Rust is not entirely a trivial code translation tasks because
// of the fact that operating on shared memory requires jumping through some hoops.

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
    Pow,
}

#[derive(Debug)]
struct ValueState {
    value: f64,
    children: Vec<Value>,
    gradient: f64,
    op: Op,
    visited: bool,
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
            visited: false,
        })))
    }

    fn from(value: f64, children: Vec<Value>, op: Op) -> Value {
        Value(Rc::new(RefCell::new(ValueState {
            value: value,
            children: children,
            gradient: 0.0,
            op: op,
            visited: false,
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
        assert!(self.0.borrow().children.len() == 2);
        self.child(0)
    }

    fn rhs(&self) -> Value {
        assert!(self.0.borrow().children.len() == 2);
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

    fn visited(&self) -> bool {
        self.0.borrow().visited
    }
}

//
// Step 2. Define Standard mathematical operations on Value objects.
//

impl Value {
    fn tanh(self) -> Value {
        Value::from(self.value().tanh(), vec![self.clone()], Op::Tanh)
    }

    fn exp(self) -> Value {
        Value::from(self.value().exp(), vec![self.clone()], Op::Exp)
    }

    fn pow(self, value: f64) -> Value {
        Value::from(self.value().powf(value), vec![self.clone()], Op::Pow)
    }
}

impl ops::Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Value::from(
            self.value() + rhs.value(),
            vec![self.clone(), rhs.clone()],
            Op::Plus,
        )
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        Value::from(
            self.value() * rhs.value(),
            vec![self.clone(), rhs.clone()],
            Op::Multiply,
        )
    }
}

impl ops::Sub<Value> for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        self + (rhs * -1.0)
    }
}

impl ops::Add<f64> for Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Value {
        self + Value::new(rhs)
    }
}

impl ops::Sub<f64> for Value {
    type Output = Value;

    fn sub(self, rhs: f64) -> Value {
        self + Value::new(-rhs)
    }
}

impl ops::Mul<f64> for Value {
    type Output = Value;

    fn mul(self, rhs: f64) -> Value {
        self * Value::new(rhs)
    }
}

impl ops::Div<Value> for Value {
    type Output = Value;

    fn div(self, rhs: Value) -> Value {
        self * rhs.pow(-1.0)
    }
}

//
// Step 3. Each Value object in a graph can know its local gradient with respect to
// the overall expression. This function calculates that gradient value.
//

impl Value {
    fn backward(&self) {
        fn _reset_children_gradients_and_visited(node: &Value) {
            for children in node.0.borrow().children.iter() {
                children.0.borrow_mut().gradient = 0.0;
                children.0.borrow_mut().visited = false;
                _reset_children_gradients_and_visited(children);
            }
        }
        _reset_children_gradients_and_visited(self);

        // Run topological sorting to compute the ordered list of parameters

        fn _find_leaf_nodes_not_visited(node: &Value) -> Vec<Value> {
            let mut result: Vec<Value> = Vec::new();
            if !node.visited() {
                let mut count = 0;
                for child in node.0.borrow().children.iter() {
                    if !child.visited() {
                        result.append(&mut _find_leaf_nodes_not_visited(child));
                        count += 1;
                    }
                }
                if count == 0 {
                    node.0.borrow_mut().visited = true;
                    result.push(node.clone())
                }
            }
            result
        }

        let mut parameters: Vec<Value> = Vec::new();
        loop {
            let mut leafs = _find_leaf_nodes_not_visited(self);
            if leafs.len() == 0 {
                break;
            }
            parameters.append(&mut leafs);
        }

        parameters.reverse();
        parameters[0].0.borrow_mut().gradient = 1.0;

        // Fill in all the gradients in reverse topological order

        for node in parameters {
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
                Op::Pow => {
                    let child_value = node.only_child().value();
                    let exponent = out_value.log10() / child_value.log10();
                    node.only_child()
                        .inc_gradient(exponent * out_value / child_value * out_gradient);
                }
                _ => (),
            }
        }
    }
}

//
// Step 4. Assemble Values into a Neuron structure that use the tanh activation function.
//

struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    fn new(inputs: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new_inclusive(-1.0, 1.0);

        let mut weights: Vec<Value> = Vec::new();
        for _ in 1..=inputs {
            weights.push(Value::new(uniform.sample(&mut rng)));
        }

        Neuron {
            weights: weights,
            bias: Value::new(uniform.sample(&mut rng)),
        }
    }

    fn forward(&self, inputs: &[f64]) -> Value {
        assert!(inputs.len() == self.weights.len());
        let mut output = self.bias.clone();
        for (input, weight) in zip(inputs, self.weights.iter()) {
            output = output + Value::new(*input) * weight.clone();
        }
        output.tanh()
    }

    fn parameters(&self) -> Vec<Value> {
        let mut result = self.weights.clone();
        result.push(self.bias.clone());
        result
    }
}

fn main() {
    let neuron = Neuron::new(2);

    let output = neuron.forward(&[1.0, 2.0]);
    output.backward();

    for p in neuron.parameters() {
        dump(&p);
    }
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
