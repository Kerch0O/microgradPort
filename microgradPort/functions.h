#pragma once
#include <iostream>
#include <ostream>
#include <set>
#include <vector>
#include <math.h>
#include "Neural.h"
#include "Value.h"


Value* mul(Value* a, Value* b);
Value* add(Value* a, Value* b);
Value* sigmoid(Value* a);
Value* tanh(Value* a);

void build_topo(Value * v, std::set<Value*>& s, std::vector<Value*>& topo);

//Activation functions
float sigmoid(const float& a);
float th(float a);

//Derivatives for activation functions
float d_tanh(const float& a);
float d_sigmoid(const float& a);