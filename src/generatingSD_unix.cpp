
/*
 *
 * Provides the functions related to UNIX operating system.
 *
 * Author:  Antoine Allard, Robert Jankowski
 * WWW:     antoineallard.info, robertjankowski.github.io/
 * Date:    November 2017, October 2023
 * 
 * To compile (from the root repository of the project):
 *   g++ -O3 -std=c++17 src/generatingSD_unix.cpp -o generatingSD_unix.out
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

#include "../include/generatingSD_unix.hpp"

int main(int argc , char *argv[])
{
  generatingSD_t the_graph;
  if(parse_options(argc, argv, the_graph))
  {
    the_graph.load_hidden_variables();
    the_graph.generate_edgelist();
  }
  return EXIT_SUCCESS;
}
