// Compute the gradient for a computational graph.

const operations = { variable: 0, constant: 1, add: 2, multiply: 3, power: 4 };

class Computation {
  constructor(value = 0.0, gradient = 0.0, operation = operations.variable, operands = []) {
    this.value = value;
    this.gradient = gradient;
    this.operation = operation;
    this.operands = operands;
  }

  add(operand) {
    if (typeof operand === 'number') {
      // If operand is a number, create a constant computation.
      operand = new Computation(operand, 0.0, operations.constant);
    }
    return new Computation(this.value + operand.value, 0.0, operations.add, [this, operand]);
  }

  negate() {
    return this.multiply(-1);
  }

  subtract(operand) {
    if (typeof operand === 'number') {
      operand = new Computation(operand, 0.0, operations.constant);
    }
    return this.add(operand.negate());
  }

  multiply(operand) {
    if (typeof operand === 'number') {
      operand = new Computation(operand, 0.0, operations.constant);
    }
    return new Computation(this.value * operand.value, 0.0, operations.multiply, [this, operand]);
  }

  divide(operand) {
    if (typeof operand === 'number') {
      operand = new Computation(operand, 0.0, operations.constant);
    }
    return this.multiply(operand.power(-1));
  }

  power(exponent) {
    if (typeof exponent === 'number') {
      exponent = new Computation(exponent, 0.0, operations.constant);
    }
    return new Computation(Math.pow(this.value, exponent.value), 0.0, operations.power, [this, exponent]);
  }

  computeGradients() {
    // This computation contains the target value.
    // We will compute d(target)/d(node) for every node in the computational graph.
    this.zerograd();
    this.gradient = 1.0;
    this.computeOperandGradients();
  }

  computeOperandGradients() {
    // For each operand, we compute d(target)/d(operand) = d(target)/d(this) * d(this)/d(operand).
    // d(target)/d(this) is this.gradient; it has already been computed.
    for (const [i, operand] of this.operands.entries()) {
      switch (this.operation) {
        case operations.variable:
        case operations.constant:
          operand.gradient += this.gradient; // d(this)/d(operand) is 1.
          break;
        case operations.add:
          operand.gradient += this.gradient; // d(a+b)/da = 1.
          break;
        case operations.multiply:
          // d(a*b)/da = b, d(a*b)/db = a.
          let product = 1.0;
          for (const [j, op] of this.operands.entries()) {
            if (i !== j) {
              product *= op.value;
            }
          }
          operand.gradient += this.gradient * product;
          break;
        case operations.power:
          if (this.operands.length !== 2) {
            throw new Error('Power operation requires exactly two operands.');
          }
          if (i === 0) {
            // d(a^b)/da = b*a^(b-1).
            operand.gradient += this.gradient * this.operands[1].value * Math.pow(operand.value, this.operands[1].value - 1);
          } else {
            // d(a^b)/db = a^b * ln(a).
            operand.gradient += this.gradient * Math.pow(operand.value, this.operands[1].value) * Math.log(operand.value);
          }
          break;
      }
      // Recursively compute gradients for operands.
      operand.computeOperandGradients();
    }
  }

  // Recursively zero all gradients in the subgraph rooted at this node.
  // Used before a new backward pass to clear stale gradients.
  zerograd() {
    this.gradient = 0.0;
    for (const operand of this.operands) {
      operand.zerograd();
    }
  }
}

class Constant extends Computation {
  constructor(value) {
    super(value, 0.0, operations.constant);
  }
}

class Variable extends Computation {
  constructor(value) {
    super(value, 0.0, operations.variable);
  }
}

// Export for testing and external usage
module.exports = { Computation, operations, Constant, Variable };
