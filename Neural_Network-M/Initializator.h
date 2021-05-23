#pragma once
class Initializator
{
public:
	enum Initializators {
		He,
		Xavier
	};
	static float He_ini(unsigned numberInputs);
	static float Xavier_ini(unsigned numberInputs, unsigned numberOutput);
};

