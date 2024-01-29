#ifndef INFER_KAPPAS_BETA_UNIX_HPP_INCLUDED
#define INFER_KAPPAS_BETA_UNIX_HPP_INCLUDED

#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>
#include "infer_kappas_beta.hpp"


void print_usage()
{
  constexpr auto usage = R"(
NAME
      Inference of hidden degrees and parameter beta from real networks

SYNOPSIS
      ./infer_kappas_beta [options] <edgelist_filename>

INPUT
      The structure of the graph is provided by a text file containing it edgelist. Each
      line in the file corresponds to an edge in the graph (i.e., [VERTEX1] [VERTEX2]).
        - The name of the vertices need not be integers (they are stored as std::string).
        - Directed graphs will be converted to undirected.
        - Multiple edges, self-loops and weights will be ignored.
        - Lines starting with '# ' are ignored (i.e., comments).
  )";
  std::cout << usage << '\n';
}

void print_help()
{
  constexpr auto help = R"(
The following options are available:
    -d [DIMENSION] Dimension of the inference.
  )";
  std::cout << help << '\n';
}

void parse_options(int argc , char *argv[], kappas_beta_inference &the_graph)
{
  // Shows the options if no argument is given.
  if(argc == 1)
  {
    print_usage();
    print_help();
    std::exit(0);
  }

  // <edgelist_filename>
  the_graph.EDGELIST_FILENAME = argv[argc - 1];

  // Parsing options.
  int opt;
  while ((opt = getopt(argc,argv,"d:")) != -1)
  {
    switch(opt)
    {
      case 'd':
        the_graph.DIMENSION = std::stoi(optarg);
        break;
      default:
        print_usage();
        print_help();
        std::exit(0);
    }
  }
}

#endif // INFER_KAPPAS_BETA_UNIX_HPP_INCLUDED
