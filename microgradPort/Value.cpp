#include "functions.h"

Value::Value(std::pair<Value*, Value*> children, const float &d) : out(nullptr) {
	data = d;
	prev = children;
}

void Value::_backward() {
	Value* other = this == out->prev.first ? out->prev.second : out->prev.first;

	switch (out->type) {
	case '+':

		grad += out->grad;
		break;
	case '*':

		grad += other->data * out->grad;
		break;
	case 's':
		//sigmoid
		grad += d_sigmoid(data) * out->grad;
		break;
	case 't':
		//tanh
		grad += d_tanh(data) * out->grad;
	}
}

void Value::backward() {
	
	std::vector<Value*> topo;
	std::set<Value*> s;
	build_topo(this, s, topo);

	grad = 1;

	for (int i = topo.size() - 2; i >= 0; i--) {
		topo[i]->_backward();
		std::cout << "Grad for " << i << ": " << topo[i]->grad << std::endl;
	}
}
