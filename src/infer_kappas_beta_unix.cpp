#include "../include/infer_kappas_beta_unix.hpp"

int main(int argc , char *argv[])
{
  // Initialize graph object.
  kappas_beta_inference the_graph;

  // Parses and sets options.
  parse_options(argc, argv, the_graph);

  // Performs the embedding. 
  the_graph.infer();
  
  // Returns successfully.
  return EXIT_SUCCESS;
}
