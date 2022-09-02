#include "functions.h"



int main() {

	Value *a = new Value(std::make_pair<Value*, Value*>(nullptr, nullptr), 1.0f);
	Value *b = new Value(std::make_pair<Value*, Value*>(nullptr, nullptr), 2.0f);

	Value* c = add(a, b); 	

	Value* d = new Value(std::make_pair<Value*, Value*>(nullptr, nullptr), -1.0f);
	Value* e = mul(c, d);
	Value* f = tanh(e);

	//I think this is all working fine :D:D
	//do more testing with real worked out values

	f->backward();
	
	

	return 0;
}