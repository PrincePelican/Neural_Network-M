#pragma once
class Layer
{
public:
	virtual void feed_forward() = 0;
	virtual void back_propagation() = 0;
};
