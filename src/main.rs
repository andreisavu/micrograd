use std::fmt;
use std::ops;

#[derive(Clone, PartialEq, Debug)]
struct Value(f64, Vec<Value>);

fn val(data: f64) -> Value {
    Value(data, Vec::new())
}

impl ops::Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Value(self.0 + rhs.0, vec![self, rhs])
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Value {
        Value(self.0 * rhs.0, vec![self, rhs])
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.1.len() > 0 {
            let last_index = self.1.len() - 1;
            write!(f, "V({}, [", self.0)?;
            for (i, child) in self.1.iter().enumerate() {
                write!(f, "{}", child)?;
                if i < last_index {
                    write!(f, ", ")?
                }
            }
            write!(f,"])")
        } else {
            write!(f, "V({})", self.0)
        }
    }
}

fn main() {
    let a = val(1.0);
    let b = val(2.0);
    let c = val(3.0);
    let d = a * b + c;
    println!("{}", d);
}
