/*
 * This class provides the functions to generate a graph in the SD space.
 *
 * Author:  Antoine Allard, Robert Jankowski
 * WWW:     antoineallard.info, robertjankowski.github.io/
 * Date:    November 2017, October 2023
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

#ifndef GENERATINGSD_HPP_INCLUDED
#define GENERATINGSD_HPP_INCLUDED

// Standard Template Library
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>



class generatingSD_t
{
  // Flags controlling options.
  public:
    bool OUTPUT_VERTICES_PROPERTIES = false;
    bool WITH_INPUT_KAPPAS = false;
  // Global parameters.
  public:
    // Random number generator seed.
    int SEED = std::time(NULL);
    // Parameter beta (controls clustering).
    double BETA = -1;
    // Parameter mu (controls average degree).
    double MU = -1;
    // Exponent of the power-law distribution for hidden degrees.
    double GAMMA = -1;
    // Network size
    int NB_VERTICES = -1;
    // Mean degree of nodes
    double MEAN_DEGREE = -1;
    // Rootname for the output files;
    std::string OUTPUT_ROOTNAME = "default_output_rootname";
    // Input hidden variables filename.
    std::string HIDDEN_VARIABLES_FILENAME = "";
    // Name of the output files
    std::string OUTPUT_FILENAME = "net";
    // Dimension of the model S^D
    int DIMENSION = 1;
  // General internal objects.

  private:
    // pi
    const double PI = 3.141592653589793238462643383279502884197;
    const double NUMERICAL_ZERO = 1e-10;
    // Random number generator
    std::mt19937 engine;
    std::uniform_real_distribution<double> uniform_01;
    std::normal_distribution<double> normal_01;
    // Mapping the numerical ID of vertices to their name.
    std::vector<std::string> Num2Name;
  // Objects related to the graph ensemble.
  private:
    // Hidden variables of the vertices.
    std::vector<double> kappa;
    // Positions of the vertices.
    std::vector<double> theta;
    // Degree = kappa
    std::vector<double> degree;
    // Position of the vertices in D-dimensions
    std::vector<std::vector<double>> d_positions;
    // Expected degrees in the inferred ensemble (analytical, no finite-size effect).
    std::vector<double> inferred_ensemble_expected_degree;
  // Public functions to generate the graphs.
  public:
    // Constructor (empty).
    generatingSD_t() {};
    // Loads the values of the hidden degrees
    void load_hidden_variables();
    // Generates an edgelist and writes it into a file.
    void generate_edgelist(int width = 15);
  private:
    // Saves the values of the hidden variables (i.e., kappa and theta).
    void save_vertices_properties(std::vector<int>& rdegree, std::vector<double>& edegree, int width);
    // Generate random coordiantes in D dimensional space
    std::vector<double> generate_random_d_vector(int dim);
    double compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2);
    inline double compute_radius(int dim, int N) const;

    void fix_error_with_zero_degree_nodes(std::vector<int>& rdegree, std::vector<double>& edegree);
    // Gets and format current date/time.
    std::string get_time();

    void validate_input_parameters();
    void generate_powerlaw_distribution();
};


void generatingSD_t::validate_input_parameters() 
{
  if (BETA <= 0) {
    throw std::invalid_argument("Parameter beta should be larger than 0.");
  }

  // Either n,gamma,mean_degree is specified or HIDDEN_VARIABLES_FILENAME is not empty
  if (!HIDDEN_VARIABLES_FILENAME.empty()) 
  {
    WITH_INPUT_KAPPAS = true;
    if (!((NB_VERTICES == -1) & (GAMMA == -1) & (MEAN_DEGREE == -1))) {
      throw std::invalid_argument("When kappas is input, n, gamma and mean_degree must not be.");
    }
  } else {
    WITH_INPUT_KAPPAS = false;
    if ((NB_VERTICES == -1) || (GAMMA == -1) || (MEAN_DEGREE == -1)) {
      throw std::invalid_argument("Please provide either kappas, or all 3 parameters: n, gamma and mean_degree.");
    }
    HIDDEN_VARIABLES_FILENAME = ".";
  }
  auto path = std::filesystem::path(HIDDEN_VARIABLES_FILENAME);
  path = std::filesystem::absolute(path);
  HIDDEN_VARIABLES_FILENAME = path.string();
  OUTPUT_ROOTNAME = path.parent_path().string() + "/" + OUTPUT_FILENAME;
}

void generatingSD_t::generate_powerlaw_distribution()
{ 
  const double kappa_0 = (1 - 1.0 / NB_VERTICES) / (1 - std::pow(NB_VERTICES, (2.0 - GAMMA) / (GAMMA - 1.0))) * (GAMMA - 2) / (GAMMA - 1) * MEAN_DEGREE;
  const double base = 1 - 1.0 / NB_VERTICES;
  const double power = 1 / (1 - GAMMA);
  
  for (int i=0; i<NB_VERTICES; ++i) { 
    auto random_kappa = kappa_0 * std::pow(1 - uniform_01(engine) * base, power);
    kappa.push_back(random_kappa);
    Num2Name.push_back("v" + std::to_string(i));
    degree.push_back(random_kappa);
  }
}

void generatingSD_t::generate_edgelist(int width)
{
  const auto inside = NB_VERTICES / (2 * std::pow(PI, (DIMENSION + 1) / 2.0)) * std::tgamma((DIMENSION + 1) / 2.0);
  const double radius = std::pow(inside, 1.0 / DIMENSION);

 // Initializes the random number generator.
  engine.seed(SEED);
  // Sets the name of the file to write the edgelist into.
  std::cout << "OUTPUT_ROOTNAME = " << OUTPUT_ROOTNAME << std::endl;
  std::string edgelist_filename = OUTPUT_ROOTNAME + ".edge";
  // Vectors containing the expected and real degrees.
  std::vector<double> edegree;
  std::vector<int> rdegree;
  // Initializes the containers for the expected and real degrees.
  if(OUTPUT_VERTICES_PROPERTIES)
  {
    edegree.resize(NB_VERTICES, 0);
    rdegree.resize(NB_VERTICES, 0);
  }
  
  // Should we keep it?
  //fix_error_with_zero_degree_nodes(rdegree, edegree);
  
  if(WITH_INPUT_KAPPAS)
  {
    // Computes the average value of kappa.
    MEAN_DEGREE = 0;
    for(int v(0); v<NB_VERTICES; ++v)
    {
      MEAN_DEGREE += kappa[v];
    }
    MEAN_DEGREE /= NB_VERTICES;
  }
  
  if (BETA > DIMENSION) {
    const auto top = BETA * std::tgamma(DIMENSION / 2.0) * std::sin(DIMENSION * PI / BETA);
    const auto bottom = MEAN_DEGREE * 2 * std::pow(PI, 1 + DIMENSION / 2.0);
    MU =  top / bottom;
  } else if (BETA < DIMENSION) {
    const auto top1 = (DIMENSION - BETA) * std::tgamma(DIMENSION / 2);
    const auto bottom1 = 2 * std::pow(PI, (3 * DIMENSION) / 2 - BETA) * MEAN_DEGREE * std::pow(NB_VERTICES, 1 - BETA / DIMENSION);
    const auto second_part = 2 * std::pow(PI, (DIMENSION + 1) / 2) / std::tgamma((DIMENSION + 1) / 2);
    MU = top1 / bottom1 * std::pow(second_part, 1 - BETA / DIMENSION);
  } else { // beta=dimension
    throw std::invalid_argument("Case beta=dimension is not implemented yet.");
  }
  
  if (DIMENSION == 1) {
    theta.clear();
    theta.resize(NB_VERTICES);
    for (int v=0; v<NB_VERTICES; ++v)
      theta[v] = 2 * PI * uniform_01(engine);
    
  } else {
    d_positions.clear();
    d_positions.resize(NB_VERTICES);
    for(int v(0); v<NB_VERTICES; ++v)
      d_positions[v] = generate_random_d_vector(DIMENSION);
  }

  // Opens the stream and terminates if the operation did not succeed.
  std::fstream edgelist_file(edgelist_filename.c_str(), std::fstream::out);
  std::cout << "edgelist_file = " << edgelist_filename << std::endl;
  if( !edgelist_file.is_open() )
  {
    std::cerr << "ERROR: Could not open file: " << edgelist_filename << "." << std::endl;
    std::terminate();
  }
  // Writes the header.
  edgelist_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=" << std::endl;
  edgelist_file << "# Generated on:           " << get_time()                << std::endl;
  edgelist_file << "# Hidden variables file:  " << HIDDEN_VARIABLES_FILENAME << std::endl;
  edgelist_file << "# Seed:                   " << SEED                      << std::endl;
  edgelist_file << "#"                                                       << std::endl;
  edgelist_file << "# Parameters"                                            << std::endl;
  edgelist_file << "#   - nb. vertices:       " << NB_VERTICES               << std::endl;
  edgelist_file << "#   - dimension:          " << DIMENSION                 << std::endl;
  edgelist_file << "#   - beta:               " << BETA                      << std::endl;
  edgelist_file << "#   - mu:                 " << MU                        << std::endl;
  edgelist_file << "#   - radius:             " << radius                    << std::endl;
  edgelist_file << "#   - gamma:              " << GAMMA                     << std::endl;
  edgelist_file << "#   - mean degree:        " << MEAN_DEGREE               << std::endl;
  edgelist_file << "# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=" << std::endl;
  edgelist_file << "#";
  edgelist_file << std::setw(width - 1) << "Vertex1" << " ";
  edgelist_file << std::setw(width)     << "Vertex2" << " ";
  edgelist_file << std::endl;
  // Generates the edgelist.
  double dtheta;
  for(int v1(0); v1<NB_VERTICES; ++v1) {
    for(int v2(v1 + 1); v2<NB_VERTICES; ++v2) {
      if (DIMENSION == 1) {
        dtheta = PI - std::fabs(PI - std::fabs(theta[v1] - theta[v2]));
      } else {
        dtheta = compute_angle_d_vectors(d_positions[v1], d_positions[v2]);
      }
      const auto inside = std::pow(radius * dtheta, BETA) / std::pow(MU * kappa[v1] * kappa[v2], std::max((double)DIMENSION, BETA) / DIMENSION);
      const auto prob = 1 / (1 + inside); 

      if(uniform_01(engine) < prob)
      {
        edgelist_file << std::setw(width) << Num2Name[v1] << " ";
        edgelist_file << std::setw(width) << Num2Name[v2] << " ";
        edgelist_file << std::endl;
        if(OUTPUT_VERTICES_PROPERTIES)
        {
          rdegree[v1] += 1;
          rdegree[v2] += 1;
        }
      }
      if(OUTPUT_VERTICES_PROPERTIES)
      {
        edegree[v1] += prob;
        edegree[v2] += prob;
      }
    }
  }
  // Closes the stream.
  edgelist_file.close();
  // Outputs the hidden variables, if required. 
  if(OUTPUT_VERTICES_PROPERTIES)
  {
    save_vertices_properties(rdegree, edegree, width);
  }
}


void generatingSD_t::load_hidden_variables()
{
  validate_input_parameters();
  if (WITH_INPUT_KAPPAS) {  
    // Stream object.
    std::stringstream one_line;
    // String objects.
    std::string full_line, name1_str, name2_str, name3_str;
    // Resets the number of vertices.
    NB_VERTICES = 0;
    // Resets the container.
    kappa.clear();
    // Opens the stream and terminates if the operation did not succeed.
    std::fstream hidden_variables_file(HIDDEN_VARIABLES_FILENAME.c_str(), std::fstream::in);
    if( !hidden_variables_file.is_open() )
    {
      std::cerr << "Could not open file: " << HIDDEN_VARIABLES_FILENAME << "." << std::endl;
      std::terminate();
    }
    // Reads the hidden variables file line by line.
    while( !hidden_variables_file.eof() )
    {
      // Reads a line of the file.
      std::getline(hidden_variables_file, full_line);
      hidden_variables_file >> std::ws;
      one_line.str(full_line);
      one_line >> std::ws;
      one_line >> name1_str >> std::ws;
      // Skips lines of comment.
      if(name1_str == "#")
      {
        one_line.clear();
        continue;
      }
      Num2Name.push_back("v" + std::to_string(NB_VERTICES));
      kappa.push_back(std::stod(name1_str));
      degree.push_back(std::stod(name1_str));    
      ++NB_VERTICES;
      one_line.clear();
    }
    // Closes the stream.
    hidden_variables_file.close();
  } else {
    generate_powerlaw_distribution();
  }
}

void generatingSD_t::save_vertices_properties(std::vector<int>& rdegree, std::vector<double>& edegree, int width)
{
  // Finds the minimal value of kappa.
  double kappa_min = *std::min_element(kappa.begin(), kappa.end());
  const auto R = compute_radius(DIMENSION, NB_VERTICES);
  const double zeta = BETA > DIMENSION ? 1 : 1 / BETA;
  const double hyp_radius = 2 / zeta * std::log(2 * R / std::pow(MU * kappa_min * kappa_min, std::max((double)DIMENSION, BETA) / (BETA * DIMENSION)));
  // Sets the name of the file to write the hidden variables into.
  std::string hidden_variables_filename = OUTPUT_ROOTNAME + ".gen_coord";
  // Opens the stream and terminates if the operation did not succeed.
  std::fstream hidden_variables_file(hidden_variables_filename.c_str(), std::fstream::out);
  if( !hidden_variables_file.is_open() )
  {
    std::cerr << "Could not open file: " << hidden_variables_filename << "." << std::endl;
    std::terminate();
  }
  // Writes the header.
  hidden_variables_file << "#";
  hidden_variables_file << std::setw(width - 1) << "Vertex"   << " ";
  hidden_variables_file << std::setw(width)     << "Kappa"    << " ";
  hidden_variables_file << std::setw(width)     << "Hyp.Rad."    << " ";
  if (DIMENSION == 1) 
  {
    hidden_variables_file << std::setw(width)     << "Theta"    << " ";
  } else {
    for (int i=0; i<DIMENSION+1; ++i)
      hidden_variables_file << std::setw(width)   << "Pos." << i << " ";  
  }
  hidden_variables_file << std::setw(width)     << "RealDeg." << " ";
  hidden_variables_file << std::setw(width)     << "Exp.Deg." << " ";
  hidden_variables_file << std::endl;
  // Writes the hidden variables.
  for(int v(0); v<NB_VERTICES; ++v)
  {
    hidden_variables_file << std::setw(width) << Num2Name[v]                                                     << " ";
    hidden_variables_file << std::setw(width) << kappa[v]                                                        << " ";
    hidden_variables_file << std::setw(width) << hyp_radius - ((2.0 * std::max((double)DIMENSION, BETA)) / (DIMENSION * BETA * zeta)) * std::log(kappa[v] / kappa_min) << " ";
    if (DIMENSION == 1) {
      hidden_variables_file << std::setw(width) << theta[v]                                                    << " ";
    } else {
      for (int i=0; i<DIMENSION+1; ++i)
        hidden_variables_file << std::setw(width)   << d_positions[v][i] << " ";
    }
    hidden_variables_file << std::setw(width) << rdegree[v]                                                     << " ";
    hidden_variables_file << std::setw(width) << edegree[v]                                                     << " ";
    hidden_variables_file << std::endl;
  }
  // Closes the stream.
  hidden_variables_file.close();
}

std::string generatingSD_t::get_time()
{
  // Gets the current date/time.
  time_t theTime = time(NULL);
  struct tm *aTime = gmtime(&theTime);
  int year    = aTime->tm_year + 1900;
  int month   = aTime->tm_mon + 1;
  int day     = aTime->tm_mday;
  int hours   = aTime->tm_hour;
  int minutes = aTime->tm_min;
  // Format the string.
  std::string the_time = std::to_string(year) + "/";
  if(month < 10)
    the_time += "0";
  the_time += std::to_string(month) + "/";
  if(day < 10)
    the_time += "0";
  the_time += std::to_string(day) + " " + std::to_string(hours) + ":";
  if(minutes < 10)
    the_time += "0";
  the_time += std::to_string(minutes) + " UTC";
  // Returns the date/time.
  return the_time;
}

std::vector<double> generatingSD_t::generate_random_d_vector(int dim) {
  std::vector<double> positions;
  positions.resize(dim + 1);
  double norm{0};
  for (auto &pos : positions) {
    pos = normal_01(engine);
    norm += pos * pos;
  }
  norm /= std::sqrt(norm);
  // Normalize vector
  for (auto &pos: positions)
    pos /= norm;

  // Rescale by the radius in a given dimension
  const auto R = compute_radius(dim, NB_VERTICES);
  for (auto &pos: positions)
    pos *= R;
  return positions;
}

double generatingSD_t::compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2) {
  double angle{0}, norm1{0}, norm2{0};
  for (int i = 0; i < v1.size(); ++i) {
    angle += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  norm1 /= sqrt(norm1);
  norm2 /= sqrt(norm2);
  
  const auto result = angle / (norm1 * norm2);
  if (std::fabs(result - 1) < NUMERICAL_ZERO)
    return 0; // the same vectors
  else
    return std::acos(result);
}

inline double generatingSD_t::compute_radius(int dim, int N) const
{
  const auto inside = N / (2 * std::pow(PI, (dim + 1) / 2.0)) * std::tgamma((dim + 1) / 2.0);
  return std::pow(inside, 1.0 / dim);
}

void generatingSD_t::fix_error_with_zero_degree_nodes(std::vector<int>& rdegree, std::vector<double>& edegree) {
  // Generated networks are usually smaller than the input ones
  // To solve this issue we propose to add N_0 nodes with kappa values 
  // sampled from the original kappas. In such a way we would obtain the 
  // network with almost the same size and average degree as the input one.
  
  double mean_exp_kappa = 0;
  for (int i=0; i<NB_VERTICES; ++i) {
    mean_exp_kappa += std::exp(-kappa[i]);
  }
  mean_exp_kappa /= NB_VERTICES;
  int N_0 = round(NB_VERTICES * mean_exp_kappa / (1 - mean_exp_kappa));  
  int new_nb_vertices = NB_VERTICES + N_0;

  std::cout << "Adding N_0 = " << N_0 << " nodes to the original network" << std::endl;  
  std::vector<double> new_kappas;
  std::sample(kappa.begin(), kappa.end(), std::back_inserter(new_kappas), N_0, std::mt19937{std::random_device{}()});

  for (const auto &k: new_kappas)
    kappa.push_back(k);

  for (int i=0; i<N_0; ++i) {
    rdegree.push_back(0);
    edegree.push_back(0);
    Num2Name.push_back("v" + std::to_string(NB_VERTICES + i));
  }
  NB_VERTICES = new_nb_vertices;
}

#endif // GENERATINGSD_HPP_INCLUDED
