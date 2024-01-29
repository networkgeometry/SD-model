/*
 *
 * Provides the functions related to UNIX operating system.
 *
 * Author:  Antoine Allard, Robert Jankowski
 * WWW:     antoineallard.info
 * Date:    November 2017, November 2023
 * 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 *
 */

#ifndef GENERATINGSD_UNIX_HPP_INCLUDED
#define GENERATINGSD_UNIX_HPP_INCLUDED

#include <cstdlib>
#include <string>
#include <unistd.h>
#include "generatingSD.hpp"


void print_usage()
{
  std::string_view message = R"""(
NAME
  generatingSD -- a program to generate complex networks in the S^D metric space

SYNOPSIS
  generatingSD [options]
  )""";
  std::cout << message << std::endl;
}

void print_help()
{
  std::string_view help = R"""(
The following options are available:
  -b [BETA]        Specifies the value for parameter beta.
  -d [DIMENSION]   Specifies model's dimension (S^D).
  -g [GAMMA]       Exponent of the power-law distribution for hidden degrees.
  -n [SIZE]        Network size.
  -k [MEAN_DEGREE] Mean degree of nodes.
  -l [KAPPAS]      File consisting of the hidden degrees 
  -s [SEED]        Program uses a custom seed for the random number generator. Default: EPOCH.
  -v               Outputs the hidden variables (kappa and nodes'positions) used to the generate the network into a file (uses the edgelist's rootname).
  -h               Print this message on screen and exit.
  -o [FILENAME]    Name of the output file (without extension) (default: net)
  )""";
  std::cout << help << std::endl;
}

bool parse_options(int argc , char *argv[], generatingSD_t &the_graph)
{
  if(argc == 1)
  {
    print_usage();
    print_help();
    return false;
  }

  // Parsing options.
  int opt;
  while ((opt = getopt(argc,argv,"b:d:g:n:k:l:hs:vo:")) != -1)
  {
    switch(opt)
    {
      case 'b':
        the_graph.BETA = std::stod(optarg);
        break;

      case 'g':
        the_graph.GAMMA = std::stod(optarg);
        break;
      
      case 'n':
        the_graph.NB_VERTICES = std::stod(optarg);
        break;

      case 'k':
        the_graph.MEAN_DEGREE = std::stod(optarg);
        break;

      case 'l':
        the_graph.HIDDEN_VARIABLES_FILENAME = optarg;
        break;

      case 's':
        the_graph.SEED = std::stoi(optarg);
        break;

      case 'v':
        the_graph.OUTPUT_VERTICES_PROPERTIES = true;
        break;
      
      case 'd':
        the_graph.DIMENSION = std::stoi(optarg);
        break;
      
      case 'o':
        the_graph.OUTPUT_FILENAME = optarg;
        break;

      case 'h':
        print_usage();
        print_help();
        return false;

      default:
        print_usage();
        print_help();
        return false;
    }
  }

  // Uses the default rootname for output files.
  size_t lastdot = the_graph.HIDDEN_VARIABLES_FILENAME.find_last_of(".");
  if(lastdot == std::string::npos)
  {
    the_graph.OUTPUT_ROOTNAME = the_graph.HIDDEN_VARIABLES_FILENAME;
  }
  the_graph.OUTPUT_ROOTNAME = the_graph.HIDDEN_VARIABLES_FILENAME.substr(0, lastdot);

  return true;
}


#endif // GENERATINGSD_UNIX_HPP_INCLUDED
