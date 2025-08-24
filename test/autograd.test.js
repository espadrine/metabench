const { test } = require('node:test');
const assert = require('node:assert/strict');
const { Variable, Constant, operations } = require('../lib/autograd');

// Basic add/multiply gradient test
test('basic add/multiply gradient', () => {
  const x = new Variable(3);
  const y = new Variable(4);
  const expr = x.add(y).multiply(y); // (x + y) * y
  expr.computeGradients();
  assert.equal(expr.gradient, 1);
  assert.equal(x.gradient, 4); // d/dx = y
  assert.equal(y.gradient, 11); // d/dy = x + 2y
});

// Constant addition test
test('constant addition', () => {
  const c = new Variable(5);
  const constFive = new Constant(5);
  const sum = c.add(constFive);
  sum.computeGradients();
  assert.equal(sum.gradient, 1);
  assert.equal(c.gradient, 1);
  assert.equal(constFive.gradient, 1);
});

// Negation test
test('negation', () => {
  const neg = new Variable(3).negate();
  neg.computeGradients();
  assert.equal(neg.gradient, 1);
  assert.equal(neg.operands[0].gradient, -1);
});

// Subtraction test
test('subtraction', () => {
  const x = new Variable(3);
  const y = new Variable(5);
  const sub = x.subtract(y);
  sub.computeGradients();
  assert.equal(sub.gradient, 1);
  assert.equal(x.gradient, 1);
  assert.equal(y.gradient, -1);
});

// Multiplication gradient test
test('multiplication gradient', () => {
  const a = new Variable(2);
  const b = new Variable(3);
  const expr = a.multiply(b); // a * b
  expr.computeGradients();
  assert.equal(expr.gradient, 1);
  assert.equal(a.gradient, 3); // d(expr)/da = b
  assert.equal(b.gradient, 2); // d(expr)/db = a
});

// Division gradient test
test('division gradient', () => {
  const a = new Variable(10);
  const b = new Variable(2);
  const expr = a.divide(b); // a / b
  expr.computeGradients();
  assert.equal(expr.gradient, 1);
  assert.equal(a.gradient, 0.5); // 1/b
  assert.equal(b.gradient, -2.5); // -a/b^2
});

// Power gradient test
test('power gradient', () => {
  const z = new Variable(2);
  const expr = z.power(3); // z^3
  expr.computeGradients();
  assert.equal(expr.gradient, 1);
  // d(expr)/dz = 3*z^2
  assert.equal(z.gradient, 12); // 3*2^2
});
