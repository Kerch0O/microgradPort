#include "functions.h"


float sigmoid(const float& a) {
	return 1.0f / (1.0f + exp(-a));
}

float th(float a) { // call th not tanh because c++ thinks im trying to link to some other stupid library :D:D:D
	float e = exp(2.0f * a);
	return (e - 1) / (e + 1);
}

float d_sigmoid(const float& a) { //Derivative of sigmoid function
	return sigmoid(a) * (1.0f - sigmoid(a));
}
float d_tanh(const float& a) {
	return 1.0f - pow(th(a), 2);
}


Value* mul(Value* a, Value* b) {
	float d = a->data * b->data;
	std::pair<Value*, Value*> t(a, b);

	Value* out = new Value(t, d);
	out->type = '*';

	a->out = out;
	b->out = out;
	
	return out;

}

Value* add(Value* a, Value* b) {
	float d = a->data + b->data;

	std::pair<Value*, Value*> t(a, b);

	Value* out = new Value(t, d);
	out->type = '+';

	a->out = out;
	b->out = out;

	return out;

}

Value* sigmoid(Value* a) {
	float d = sigmoid(a->data);

	std::pair<Value*, Value*> t(a, nullptr);

	Value* out = new Value(t, d);
	out->type = 's';

	a->out = out;

	return out;
}

Value* tanh(Value* a) {
	float d = th(a->data);
	std::pair<Value*, Value*> t(a, nullptr);

	Value* out = new Value(t, d);
	out->type = 't';

	a->out = out;

	return out;
}

void build_topo(Value* v, std::set<Value*> &s, std::vector<Value*> &topo) {
	
	s.insert(v);

	if (s.find(v->prev.first) == s.end() && v->prev.first != nullptr) {
		build_topo(v->prev.first, s, topo);
	}
	if (s.find(v->prev.second) == s.end() && v->prev.second != nullptr) {
		build_topo(v->prev.second, s, topo);
	}

	topo.push_back(v);
}

