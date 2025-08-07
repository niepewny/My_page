#include "ConfigLoader.h"
#include "DatasetGenerator.h"

int main(int argc, char* argv[])
{
	std::string configPath = (argc > 1) ? argv[1] : "config.json";
	Config cfg = loadConfig(configPath);
	DatasetGenerator generator(cfg);
	generator.run();
}