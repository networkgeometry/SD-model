#ifndef INFER_KAPPAS_BETA_HPP
#define INFER_KAPPAS_BETA_HPP

#include <algorithm>
#include <cmath>
#include <ctime>
#include <complex>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
// Custom library for specific Gaussian hypergeometric functions.
#include "hyp2f1.hpp"
#include "integrate_expected_degree.hpp"


class kappas_beta_inference
{
  private:
    const double PI = 3.141592653589793238462643383279502884197;
  public:
    // Has a custom value for beta been provided?
    bool CUSTOM_BETA = false;
    // Does not provide any information during the embedding process.
    bool QUIET_MODE = false;
    // Print information about the embedding process on screen instead than on the log file.
    bool VERBOSE_MODE = false;

    int DIMENSION = 2;
  
  public:
    // Name of the file containing the previously inferred parameters.
    std::string ALREADY_INFERRED_PARAMETERS_FILENAME;
    // Minimal/maximal value of beta that the program can handle (bounds).
    const double BETA_ABS_MAX = 25;
    const double BETA_ABS_MIN = 1.01;
    // Edgelist filename.
    std::string EDGELIST_FILENAME;
    // Number of points for MC integration in the calculation of expected clustering.
    const int EXP_CLUST_NB_INTEGRATION_MC_STEPS = 600;
    // Maximal number of attempts to reach convergence of the updated values of kappa.
    const int KAPPA_MAX_NB_ITER_CONV =  500;
    const int KAPPA_MAX_NB_ITER_CONV_2 = 500;

    // Various numerical/convergence thresholds.
    const double NUMERICAL_CONVERGENCE_THRESHOLD_1 = 1e-2;
    const double NUMERICAL_CONVERGENCE_THRESHOLD_2 = 5e-5;
    const double NUMERICAL_CONVERGENCE_THRESHOLD_3 = 0.5;
    const double NUMERICAL_ZERO = 1e-10;
    // Rootname of output files.
    std::string ROOTNAME_OUTPUT;
    // Random number generator seed.
    int SEED;
  private:
    // Version of the code.
    std::string_view VERSION = "1.0";
    // Tab.
    std::string_view TAB = "    ";

  // General internal objects.
  private:
    // Random number generator
    std::mt19937 engine;
    std::uniform_real_distribution<double> uniform_01;
    std::normal_distribution<double> normal_01;
    // Objects mapping the name and the numerical ID of vertices.
    std::map< std::string, int > Name2Num;
    std::vector<std::string> Num2Name;
    // List of degree classes.
    std::set<int> degree_class;
    // Cumulative probability used for the calculation of clustering using MC integration.
    std::map<int, std::map<double, int, std::less<>>> cumul_prob_kgkp;
    // List containing the order in which the vertices will be considered in the maximization phase.
    std::vector<int> ordered_list_of_vertices;
    // Time stamps.
    time_t time_started;
    // Stream used to output the log to file.
    std::ofstream logfile;
    std::streambuf *old_rdbuf;
    // Widths of the columns in output file.
    int width_names;
    int width_values;

  // Objects related to the original graph.
  private:
    // Number of vertices.
    int nb_vertices;
    int nb_vertices_degree_gt_one;
    // Number of edges.
    int nb_edges;
    // Average degree.
    double average_degree;
    // Average local clustering coefficient.
    double average_clustering;
    // Average degree of neighbors.
    std::vector<double> sum_degree_of_neighbors;
    // Local clustering.
    std::vector<double> nbtriangles;
    // Adjacency list.
    std::vector< std::set<int> > adjacency_list;
    // Degree.
    std::vector<int> degree;
    // ID of vertices in each degree class.
    std::map<int, std::vector<int> > degree2vertices;

  // Objects related to the inferred graph ensemble.
  public:
    // Parameter beta (clustering).
    double beta;
  private:
    // Average degree of the inferred ensemble.
    double random_ensemble_average_degree;
    // Average local clustering coefficient of the inferred ensemble.
    double random_ensemble_average_clustering;
    // Maps containing the expected degree of each degree class.
    std::map<int, double> random_ensemble_expected_degree_per_degree_class;
    // Expected degrees in the inferred ensemble (analytical, no finite-size effect).
    std::vector<double> inferred_ensemble_expected_degree;
    // List of kappas by degree class.
    std::map<int, double> random_ensemble_kappa_per_degree_class;
    // Parameter mu (average degree).
    double mu;
    // Hidden variables of the vertices.
    std::vector<double> kappa;
    // Positions of the vertices.
    std::vector<double> theta;
    // Positions of the vertices in S^D
    std::vector<std::vector<double>> d_positions;
    
  // Internal functions.
  private:
    // Extracts all relevant information about the degrees.
    void analyze_degrees();
    // Computes the average local clustering coefficient.
    void compute_clustering();
    // Loads the graph from an edgelist in a file.
    void load_edgelist();
    // Builds the cumulative distribution to choose degree classes in the calculation of clustering.
    void build_cumul_dist_for_mc_integration();
    void build_cumul_dist_for_mc_integration(int dim);
    // Computes various properties of the random ensemble (before finding optimal positions).
    void compute_random_ensemble_average_degree();
    void compute_random_ensemble_clustering(int dim);
    void compute_random_ensemble_clustering();
    double compute_random_ensemble_clustering_for_degree_class(int d1, int dim);
    double compute_random_ensemble_clustering_for_degree_class(int d1);
    void infer_kappas_given_beta_for_degree_class();
    void infer_kappas_given_beta_for_degree_class(int dim);
    
    // Loads the network and computes topological properties.
    void initialize();
    // Infers the parameters (kappas, beta).
    void infer_parameters();
    // Infers the parameters (kappas, beta).
    void infer_parameters(int dim);
    // Extracts the onion decomposition (OD) of the graph and orders the vertices according to it.
    void order_vertices();
    // Updates the value of the expected degrees given the inferred positions of theta.
    void compute_inferred_ensemble_expected_degrees();
    void compute_inferred_ensemble_expected_degrees(int dim, double radius);
    // Extracts the onion decomposition.
    void extract_onion_decomposition(std::vector<int> &coreness, std::vector<int> &od_layer);
    // Gets and format current date/time.
    std::string format_time(time_t _time);
    
    // Function associated to the extraction of the components.
    int get_root(int i, std::vector<int> &clust_id);
    void merge_clusters(std::vector<int> &size, std::vector<int> &clust_id);
    void check_connected_components();
    // Calculate mu which controls average degree
    inline double calculateMu() const;
    // Gets the degree of the random vertex and computes probability of being connected
    std::pair<int, double> degree_of_random_vertex_and_prob_conn(int d1, double R);
    std::pair<int, double> degree_of_random_vertex_and_prob_conn(int d1, double R, int dim);
    // Draw random angular distance between nodes with degree d1 and d2, which are connected with probability p12
    double draw_random_angular_distance(int d1, int d2, double R, double p12);
    double draw_random_angular_distance(int d1, int d2, double R, double p12, int dim);
    // Compute the radius in S^D model with a given network size.
    inline double compute_radius(int dim, int N) const;
    // Calculate mu in D-dimension
    inline double calculate_mu(int dim) const;
    // Generate random normalized vector in D+1 dimensional space
    std::vector<double> generate_random_d_vector(int dim, double radius);
    // Generate random normalized vector in D+1 dimensional space with the first coordinate: x1 = Cos[angle] (after normalization)
    std::vector<double> generate_random_d_vector_with_first_coordinate(int dim, double angle, double radius);
    // Compute angle between two vectors in D+1 dimensional space
    double compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2);
    // Normalize vector
    void normalize_and_rescale_vector(std::vector<double> &v, double radius);
    // Save hidden degrees and exit
    void save_kappas_and_exit();
  // Public functions to perform the embeddings.
  public:
    // Constructor (empty).
    kappas_beta_inference() {};
    // Destructor (empty).
    ~kappas_beta_inference() {};
    // Performs the embedding.
    void infer();
    void infer(int dim); // Perform the embedding in D dimension
    void infer(std::string edgelist_filename) { EDGELIST_FILENAME = edgelist_filename; infer(); };
};


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::analyze_degrees()
{
  // Resets the value of the average degree.
  average_degree = 0;
  // Resets the number of vertices with a degree greater than 1.
  nb_vertices_degree_gt_one = 0;
  // Resizes the list of degrees.
  degree.clear();
  degree.resize(nb_vertices);
  // Populates the list of degrees, the average degree and
  //   the list of vertices of each degree class.
  std::set<int>::iterator it, end;
  for(int n(0), k; n<nb_vertices; ++n)
  {
    k = adjacency_list[n].size();
    degree[n] = k;
    average_degree += k;
    degree2vertices[k].push_back(n);
    degree_class.insert(k);
    if(k > 1)
    {
      ++nb_vertices_degree_gt_one;
    }
  }
  // Completes the computation of the average degree.
  average_degree /= nb_vertices;
}

void kappas_beta_inference::build_cumul_dist_for_mc_integration(int dim) {
  int v1;
  double tmp_val, tmp_cumul;
  const double R = compute_radius(dim, nb_vertices);
  mu = calculate_mu(dim);
  // Temporary container.
  std::map<int, double> nkkp;
  // Resets the main object.
  cumul_prob_kgkp.clear();
  // Iterator objects.
  std::set<int>::iterator it1, end1, it2, end2;
  // For all degree classes over 1.
  it1 = degree_class.begin();
  end1 = degree_class.end();
  while(*it1 < 2) { ++it1; }
  for(; it1!=end1; ++it1)
  {
    // Reinitializes the temporary container.
    nkkp.clear();

    // For all degree classes.
    it2 = degree_class.begin();
    end2 = degree_class.end();
    for(; it2!=end2; ++it2) {
      // Initializes the temporary container.
      nkkp[*it2] = 0;
    }

    // For all degree classes.
    it2 = degree_class.begin();
    end2 = degree_class.end();
    for(; it2!=end2; ++it2) {
      const auto kappa1 = random_ensemble_kappa_per_degree_class[*it1];
      const auto kappa2 = random_ensemble_kappa_per_degree_class[*it2];
      tmp_val = compute_integral_expected_degree_dimensions(dim, R, mu, beta, kappa1, kappa2);
      nkkp[*it2] = degree2vertices[*it2].size() * tmp_val / random_ensemble_expected_degree_per_degree_class[*it1];
    }

    // Initializes the cumulating variable.
    tmp_cumul = 0;
    // Initializes the sub-container.
    cumul_prob_kgkp[*it1];
    // For all degree classes.
    it2 = degree_class.begin();
    end2 = degree_class.end();
    for(; it2!=end2; ++it2) {
      tmp_val = nkkp[*it2];
      if(tmp_val > NUMERICAL_ZERO)
      {
        // Cumulates the probabilities;
        tmp_cumul += tmp_val;
        // Builds the cumulative distribution.
        cumul_prob_kgkp[*it1][tmp_cumul] = *it2;
      }
    }
  }
}

// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::build_cumul_dist_for_mc_integration()
{
  // Variables.
  int v1;
  double tmp_val, tmp_cumul;
  // Parameters.
  double R = nb_vertices / (2 * PI);
  mu = calculateMu();
  // Temporary container.
  std::map<int, double> nkkp;
  // Resets the main object.
  cumul_prob_kgkp.clear();
  // Iterator objects.
  std::set<int>::iterator it1, end1, it2, end2;
  // For all degree classes over 1.
  it1 = degree_class.begin();
  end1 = degree_class.end();
  while(*it1 < 2) { ++it1; }
  for(; it1!=end1; ++it1)
  {
    // Reinitializes the temporary container.
    nkkp.clear();

    // For all degree classes.
    it2 = degree_class.begin();
    end2 = degree_class.end();
    for(; it2!=end2; ++it2)
    {
      // Initializes the temporary container.
      nkkp[*it2] = 0;
    }

    // For all degree classes.
    it2 = degree_class.begin();
    end2 = degree_class.end();
    for(; it2!=end2; ++it2)
    {
      tmp_val = hyp2f1a(beta, -std::pow((PI * R) / (mu * random_ensemble_kappa_per_degree_class[*it1] * random_ensemble_kappa_per_degree_class[*it2]), beta));
      nkkp[*it2] = degree2vertices[*it2].size() * tmp_val / random_ensemble_expected_degree_per_degree_class[*it1];
    }

    // Initializes the cumulating variable.
    tmp_cumul = 0;
    // Initializes the sub-container.
    cumul_prob_kgkp[*it1];
    // For all degree classes.
    it2 = degree_class.begin();
    end2 = degree_class.end();
    for(; it2!=end2; ++it2)
    {

      tmp_val = nkkp[*it2];
      if(tmp_val > NUMERICAL_ZERO)
      {
        // Cumulates the probabilities;
        tmp_cumul += tmp_val;
        // Builds the cumulative distribution.
        cumul_prob_kgkp[*it1][tmp_cumul] = *it2;
      }
    }
  }
}


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::compute_clustering()
{
  average_clustering = 0;
  // Variables.
  double nb_triangles, tmp_val;
  // Vector objects.
  std::vector<int> intersection;
  // Set objects.
  std::set<int> neighbors_v2;
  // Iterator objects.
  std::vector<int>::iterator it;
  std::set<int>::iterator it1, end1, it2, end2;
  std::map<int, std::vector<int> >::iterator it3, end3;
  // Computes the intersection for the in- and out- neighbourhoods of each vertex.
  for(int v1(0), d1; v1<nb_vertices; ++v1)
  {
    // Resets the local clustering coefficient.
    nb_triangles = 0;
    // Performs the calculation only if degree > 1.
    d1 = degree[v1];
    if( d1 > 1 )
    {
      // Loops over the neighbors of vertex v1.
      it1 = adjacency_list[v1].begin();
      end1 = adjacency_list[v1].end();
      for(; it1!=end1; ++it1)
      {
        // Performs the calculation only if degree > 1.
        if( degree[*it1] > 1 )
        {
          // Builds an ordered list of the neighbourhood of v2
          it2 = adjacency_list[*it1].begin();
          end2 = adjacency_list[*it1].end();
          neighbors_v2.clear();
          for(; it2!=end2; ++it2)
          {
            if(*it1 < *it2) // Ensures that triangles will be counted only once.
            {
              neighbors_v2.insert(*it2);
            }
          }
          // Counts the number of triangles.
          if(neighbors_v2.size() > 0)
          {
            intersection.clear();
            intersection.resize(std::min(adjacency_list[v1].size(), neighbors_v2.size()));
            it = std::set_intersection(adjacency_list[v1].begin(), adjacency_list[v1].end(),
                                  neighbors_v2.begin(), neighbors_v2.end(), intersection.begin());
            intersection.resize(it-intersection.begin());
            nb_triangles += intersection.size();
          }
        }
      }
      // Adds the contribution of vertex v1 to the average clustering coefficient.
      tmp_val = 2 * nb_triangles / (d1 * (d1 - 1));
      average_clustering += tmp_val;
    }
  }
  // Completes the calculation of the average local clustering coefficient.
  average_clustering /= nb_vertices_degree_gt_one;
}

void kappas_beta_inference::compute_inferred_ensemble_expected_degrees(int dim, double radius)
{
  // Computes the new expected degrees given the inferred values of theta.
  inferred_ensemble_expected_degree.clear();
  inferred_ensemble_expected_degree.resize(nb_vertices, 0);
  for(int v1=0; v1<nb_vertices; ++v1) {
    for(int v2(v1 + 1); v2<nb_vertices; ++v2) {
      const auto dtheta = compute_angle_d_vectors(d_positions[v1], d_positions[v2]);
      const auto chi = radius * dtheta / std::pow(mu * kappa[v1] * kappa[v2], 1.0 / dim);
      const auto prob = 1 / (1 + std::pow(chi, beta));
      inferred_ensemble_expected_degree[v1] += prob;
      inferred_ensemble_expected_degree[v2] += prob;  
    }
  }
}

// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::compute_inferred_ensemble_expected_degrees()
{
  // Variables.
  double kappa1, theta1, dtheta, prob;
  double prefactor = nb_vertices / (2 * PI * mu);
  // Computes the new expected degrees given the inferred values of theta.
  inferred_ensemble_expected_degree.clear();
  inferred_ensemble_expected_degree.resize(nb_vertices, 0);
  for(int v1(0); v1<nb_vertices; ++v1)
  {
    kappa1 = kappa[v1];
    theta1 = theta[v1];
    for(int v2(v1 + 1); v2<nb_vertices; ++v2)
    {
      dtheta = PI - std::fabs(PI - std::fabs(theta1 - theta[v2]));
      prob = 1 / (1 + std::pow((prefactor * dtheta) / (kappa1 * kappa[v2]), beta));
      inferred_ensemble_expected_degree[v1] += prob;
      inferred_ensemble_expected_degree[v2] += prob;
    }
  }
}


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::compute_random_ensemble_average_degree()
{
  // Computes the ensemble average degree.
  random_ensemble_average_degree = 0;
  for (const auto &it2: random_ensemble_expected_degree_per_degree_class)
    random_ensemble_average_degree += it2.second * degree2vertices[it2.first].size();
  random_ensemble_average_degree /= nb_vertices;
}


void kappas_beta_inference::compute_random_ensemble_clustering(int dim)
{
  // Reinitializes the average clustering for the ensemble.
  random_ensemble_average_clustering = 0;
  // Computes the inferred ensemble clustering spectrum for all degree classes over 1.
  auto it = degree_class.begin();
  auto end = degree_class.end();
  while(*it < 2) { ++it; }
  for(; it!=end; ++it)
  {
    // Computes the clustering coefficient for the degree class and updates the average value.
    double p23 = compute_random_ensemble_clustering_for_degree_class(*it, dim);
    random_ensemble_average_clustering += p23 * degree2vertices[*it].size();
  }
  // Completes the calculation of the average clustering coefficient of the inferred ensemble.
  random_ensemble_average_clustering /= nb_vertices_degree_gt_one;
}

// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::compute_random_ensemble_clustering()
{
  // Reinitializes the average clustering for the ensemble.
  random_ensemble_average_clustering = 0;
  // Computes the inferred ensemble clustering spectrum for all degree classes over 1.
  auto it = degree_class.begin();
  auto end = degree_class.end();
  while(*it < 2) { ++it; }
  for(; it!=end; ++it)
  {
    // Computes the clustering coefficient for the degree class and updates the average value.
    double p23 = compute_random_ensemble_clustering_for_degree_class(*it);
    // ensemble_clustering_spectrum[*it] = p23;
    random_ensemble_average_clustering += p23 * degree2vertices[*it].size();
  }
  // Completes the calculation of the average clustering coefficient of the inferred ensemble.
  random_ensemble_average_clustering /= nb_vertices_degree_gt_one;
}

std::pair<int, double> kappas_beta_inference::degree_of_random_vertex_and_prob_conn(int d1, double R, int dim)
{
  auto d = cumul_prob_kgkp[d1].lower_bound(uniform_01(engine))->second;
  const auto kappa1 = random_ensemble_kappa_per_degree_class[d1];
  const auto kappa2 = random_ensemble_kappa_per_degree_class[d];
  auto p = compute_integral_expected_degree_dimensions(dim, R, mu, beta, kappa1, kappa2);
  return std::make_pair(d, p);
}

std::pair<int, double> kappas_beta_inference::degree_of_random_vertex_and_prob_conn(int d1, double R)
{
  auto d = cumul_prob_kgkp[d1].lower_bound(uniform_01(engine))->second;
  auto p = hyp2f1a(beta, -std::pow((PI * R) / (mu * random_ensemble_kappa_per_degree_class[d1] * random_ensemble_kappa_per_degree_class[d]), beta));
  return std::make_pair(d, p);
}

double kappas_beta_inference::draw_random_angular_distance(int d1, int d2, double R, double p12, int dim)
{
  double pc = uniform_01(engine);
  double zmin = 0, zmax = PI, z, pz;
  const auto kappa1 = random_ensemble_kappa_per_degree_class[d1];
  const auto kappa2 = random_ensemble_kappa_per_degree_class[d2];
  while((zmax - zmin) > NUMERICAL_CONVERGENCE_THRESHOLD_2)
  {
    z = (zmax + zmin) / 2;
    pz = compute_integral_expected_degree_dimensions(dim, R, mu, beta, kappa1, kappa2, z) / p12;
    if(pz > pc)
      zmax = z;
    else
      zmin = z;
  }
  return (zmax + zmin) / 2;
}

double kappas_beta_inference::draw_random_angular_distance(int d1, int d2, double R, double p12)
{
  double pc = uniform_01(engine);
  double zmin = 0, zmax = PI, z, pz;
  while((zmax - zmin) > NUMERICAL_CONVERGENCE_THRESHOLD_2)
  {
    z = (zmax + zmin) / 2;
    pz = (z / PI) * hyp2f1a(beta, -std::pow((z * R) / (mu * random_ensemble_kappa_per_degree_class[d1] * random_ensemble_kappa_per_degree_class[d2]), beta)) / p12;
    if(pz > pc)
      zmax = z;
    else
      zmin = z;
  }
  return (zmax + zmin) / 2;
}

// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
double kappas_beta_inference::compute_random_ensemble_clustering_for_degree_class(int d1, int dim)
{
  // Variables.
  double z12, z13, da;
  // Parameters.
  double p23 = 0;
  const int nb_points = EXP_CLUST_NB_INTEGRATION_MC_STEPS;
  const double R = compute_radius(dim, nb_vertices);
  mu = calculate_mu(dim);
  // MC integration.
#pragma omp parallel for default(shared) reduction(+:p23)
  for(int i=0; i<nb_points; ++i)
  {
    // Gets the degree of vertex 2 and 3; and Computes their probability of being connected (A.3.2.i).
    const auto [d2, p12] = degree_of_random_vertex_and_prob_conn(d1, R, dim);
    const auto [d3, p13] = degree_of_random_vertex_and_prob_conn(d1, R, dim);

    // Random angular distances between vertex (1, 2) and (1, 3) (A.3.2.ii).
    z12 = draw_random_angular_distance(d1, d2, R, p12, dim);
    z13 = draw_random_angular_distance(d1, d3, R, p13, dim);

    // Draw two random D+1 vector with first coordinate constrained by angle and compute distance between them
    const auto v1 = generate_random_d_vector_with_first_coordinate(dim, z12, R);
    const auto v2 = generate_random_d_vector_with_first_coordinate(dim, z13, R);
    const auto d_angle = compute_angle_d_vectors(v1, v2);
    if (d_angle < NUMERICAL_ZERO) {
      p23 += 1;
    } else {
      const auto kappa1 = random_ensemble_kappa_per_degree_class[d2];
      const auto kappa2 = random_ensemble_kappa_per_degree_class[d3];
      const auto inside = (R * d_angle / std::pow(mu * kappa1 * kappa2, 1.0 / dim));
      p23 += 1.0 / (1 + std::pow(inside, beta));
    }
  }
  // Returns the value of the local clustering coefficient for this degree class (A.3.2.iv).
  return p23 / nb_points;
}


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
double kappas_beta_inference::compute_random_ensemble_clustering_for_degree_class(int d1)
{
  // Variables.
  double z12, z13, da;
  double p23 = 0;
  // Parameters.
  int nb_points = EXP_CLUST_NB_INTEGRATION_MC_STEPS;
  const double R = nb_vertices / (2 * PI);
  mu = calculateMu();
  // MC integration.
#pragma omp parallel for default(shared) reduction(+:p23)
  for(int i=0; i<nb_points; ++i)
  {
    // Gets the degree of vertex 2 and 3; and Computes their probability of being connected (A.3.2.i).
    const auto [d2, p12] = degree_of_random_vertex_and_prob_conn(d1, R);
    const auto [d3, p13] = degree_of_random_vertex_and_prob_conn(d1, R);

    // Random angular distances between vertex (1, 2) and (1, 3) (A.3.2.ii).
    z12 = draw_random_angular_distance(d1, d2, R, p12);
    z13 = draw_random_angular_distance(d1, d3, R, p13);

    // Set the angular distances (A.3.2.iii)
    if(uniform_01(engine) < 0.5)
      da = std::fabs(z12 + z13);
    else
      da = std::fabs(z12 - z13);

    da = std::min(da, (2.0 * PI) - da);
    if(da < NUMERICAL_ZERO)
      p23 += 1;
    else
      p23 += 1.0 / (1.0 + std::pow((da * R) / (mu * random_ensemble_kappa_per_degree_class[d2] * random_ensemble_kappa_per_degree_class[d3]), beta));
  }
  // Returns the value of the local clustering coefficient for this degree class (A.3.2.iv).
  return p23 / nb_points;
}

// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::infer()
{
  if (DIMENSION > 1) {
    infer(DIMENSION);
    exit(0);
  }

  initialize();
  infer_parameters();
  std::cout << "Inferred beta = " << beta << std::endl;
  save_kappas_and_exit();
}

void kappas_beta_inference::infer(int dim)
{
  initialize();
  infer_parameters(dim);
  std::cout << "Inferred beta = " << beta << std::endl;
  save_kappas_and_exit();
}


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::extract_onion_decomposition(std::vector<int> &coreness, std::vector<int> &od_layer)
{
  // Builds two lists (std::vector, std::set) of the degree of the vertices.
  std::vector<int> DegreeVec(nb_vertices);
  std::set<std::pair<int, int> > DegreeSet;
  for(int v(0); v<nb_vertices; ++v)
  {
    DegreeSet.insert(std::make_pair(degree[v], v));
    DegreeVec[v] = degree[v];
  }

  // Determines the coreness and the layer based on the modified algorithm of Batagelj and
  //   Zaversnik by Hébert-Dufresne, Grochow and Allard.
  int v1, v2, d1, d2;
  int current_layer = 0;
  // int current_core = 0;
  std::set<int>::iterator it1, end;
  std::set< std::pair<int, int> > LayerSet;
  // std::set< std::pair<int, int> > order_in_layer;
  std::set< std::pair<int, int> >::iterator m_it;
  // std::set< std::pair<int, int> >::iterator o_it, o_end;
  while(!DegreeSet.empty())
  {
    // Populates the set containing the vertices belonging to the same layer.
    m_it = DegreeSet.begin();
    d1 = m_it->first;
    // Increases the layer id.
    current_layer += 1;
    // Sets the coreness and the layer the vertices with the same degree.
    while(m_it->first == d1 && m_it != DegreeSet.end())
    {
      // Sets the coreness and the layer.
      v1 = m_it->second;
      coreness[v1] = d1;
      od_layer[v1] = current_layer;
      // Looks at the next vertex.
      ++m_it;
    }
    // Adds the vertices of the layer to the set.
    LayerSet.insert(DegreeSet.begin(), m_it);
    // Removes the vertices of the current layer.
    DegreeSet.erase(DegreeSet.begin(), m_it);
    // Modifies the "effective" degree of the neighbors of the vertices in the layer.
    while(!LayerSet.empty())
    {
      // Gets information about the next vertex of the layer.
      v1 = LayerSet.begin()->second;
      // Reduces the "effective" degree of its neighbours.
      it1 = adjacency_list[v1].begin();
      end = adjacency_list[v1].end();
      for(; it1!=end; ++it1)
      {
        // Identifies the neighbor.
        v2 = *it1;
        d2 = DegreeVec[v2];
        // Finds the neighbor in the list "effective" degrees.
        m_it = DegreeSet.find(std::make_pair(d2, v2));
        if(m_it != DegreeSet.end())
        {
          if(d2 > d1)
          {
            DegreeVec[v2] = d2 - 1;
            DegreeSet.erase(m_it);
            DegreeSet.insert(std::make_pair(d2 - 1, v2));
          }
        }
      }
      // Removes the vertices from the LayerSet.
      LayerSet.erase(LayerSet.begin());
    }
  }
}


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
std::string kappas_beta_inference::format_time(time_t _time)
{
  // Gets the current date/time.
  struct tm *aTime = gmtime(& _time);
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

void kappas_beta_inference::infer_kappas_given_beta_for_degree_class(int dim)
{
  // Variable.
  double prob_conn;
  const auto radius = compute_radius(dim, nb_vertices);
  mu = calculate_mu(dim);
  // Iterators.
  std::set<int>::iterator it1, it2, end;
  // Initializes the kappas for each degree class.
  it1 = degree_class.begin();
  end = degree_class.end();
  // 1. Initialize
  for(; it1!=end; ++it1)
  {
    random_ensemble_kappa_per_degree_class[*it1] = *it1;
  }

  // 2. Finds the values of kappa generating the degree classes, given the parameters.
  int cnt = 0;
  bool keep_going = true;
  while (keep_going && (cnt < KAPPA_MAX_NB_ITER_CONV_2))
  {
    // Initializes the expected degree of each degree class.
    it1 = degree_class.begin();
    end = degree_class.end();
    for(; it1!=end; ++it1)
    {
      random_ensemble_expected_degree_per_degree_class[*it1] = 0;
    }
    // Computes the expected degrees given the actual kappas.
    //it1 = degree_class.begin();
    end = degree_class.end();
    for(it1=degree_class.begin(); it1!=end; ++it1)
    // std::for_each(std::execution::seq, degree_class.begin(), degree_class.end(), [&](const auto &it1)
    {
      it2 = it1;
      auto kappa_i = random_ensemble_kappa_per_degree_class[*it1];
      auto kappa_j = random_ensemble_kappa_per_degree_class[*it2];
      auto integral = compute_integral_expected_degree_dimensions(dim, radius, mu, beta, kappa_i, kappa_j);
      prob_conn = integral;
     
      random_ensemble_expected_degree_per_degree_class[*it1] += prob_conn * (degree2vertices[*it2].size() - 1);
      for(++it2; it2!=end; ++it2)
      {
        kappa_i = random_ensemble_kappa_per_degree_class[*it1];
        kappa_j = random_ensemble_kappa_per_degree_class[*it2];
        integral = compute_integral_expected_degree_dimensions(dim, radius, mu, beta, kappa_i, kappa_j);
        prob_conn = integral;
        random_ensemble_expected_degree_per_degree_class[*it1] += prob_conn * degree2vertices[*it2].size();
        random_ensemble_expected_degree_per_degree_class[*it2] += prob_conn * degree2vertices[*it1].size();
      }
    }
    // Verifies convergence.
    keep_going = false;
    it1 = degree_class.begin();
    end = degree_class.end();
    for(; it1!=end; ++it1)
    {
      if(std::fabs(random_ensemble_expected_degree_per_degree_class[*it1] - *it1) > NUMERICAL_CONVERGENCE_THRESHOLD_1) {
        keep_going = true;
        break;
      }
    }
    // Modifies the value of the kappas prior to the next iteration, if required.
    if(keep_going)
    {
      it1 = degree_class.begin();
      end = degree_class.end();
      for(; it1!=end; ++it1) {
        random_ensemble_kappa_per_degree_class[*it1] += (*it1 - random_ensemble_expected_degree_per_degree_class[*it1]) * uniform_01(engine);
        random_ensemble_kappa_per_degree_class[*it1] = std::fabs(random_ensemble_kappa_per_degree_class[*it1]);
      }
    }
    ++cnt;
    }
    if (cnt >= KAPPA_MAX_NB_ITER_CONV_2) {
      if (!QUIET_MODE) {
        std::clog << std::endl;
        std::clog << TAB << "WARNING: maximum number of iterations reached before convergence. This limit can be"  << std::endl;
        std::clog << TAB << "         adjusted by setting the parameters KAPPA_MAX_NB_ITER_CONV_2 to desired value." << std::endl;
        std::clog << TAB << std::fixed << std::setw(11) << " " << " ";
      }
    }
}

// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::infer_kappas_given_beta_for_degree_class()
{
  // Variable.
  double prob_conn;
  // Parameters.
  mu = calculateMu();
  // Iterators.
  std::set<int>::iterator it1, it2, end;
  // Initializes the kappas for each degree class.
  it1 = degree_class.begin();
  end = degree_class.end();
  for(; it1!=end; ++it1)
  {
    random_ensemble_kappa_per_degree_class[*it1] = *it1;
  }

  // Finds the values of kappa generating the degree classes, given the parameters.
  int cnt = 0;
  bool keep_going = true;
  while( keep_going && (cnt < KAPPA_MAX_NB_ITER_CONV) )
  {
    // Initializes the expected degree of each degree class.
    it1 = degree_class.begin();
    end = degree_class.end();
    for(; it1!=end; ++it1)
    {
      random_ensemble_expected_degree_per_degree_class[*it1] = 0;
    }
    // Computes the expected degrees given the actual kappas.
    it1 = degree_class.begin();
    end = degree_class.end();
    for(; it1!=end; ++it1)
    {
      it2 = it1;
      prob_conn = hyp2f1a(beta, -std::pow(nb_vertices / (2.0 * mu * random_ensemble_kappa_per_degree_class[*it1] * random_ensemble_kappa_per_degree_class[*it2]), beta));
      random_ensemble_expected_degree_per_degree_class[*it1] += prob_conn * (degree2vertices[*it2].size() - 1);
      for(++it2; it2!=end; ++it2)
      {
        prob_conn = hyp2f1a(beta, -std::pow(nb_vertices / (2.0 * mu * random_ensemble_kappa_per_degree_class[*it1] * random_ensemble_kappa_per_degree_class[*it2]), beta));
        random_ensemble_expected_degree_per_degree_class[*it1] += prob_conn * degree2vertices[*it2].size();
        random_ensemble_expected_degree_per_degree_class[*it2] += prob_conn * degree2vertices[*it1].size();
      }
    }
    // Verifies convergence.
    keep_going = false;
    it1 = degree_class.begin();
    end = degree_class.end();
    for(; it1!=end; ++it1)
    {
      if(std::fabs(random_ensemble_expected_degree_per_degree_class[*it1] - *it1) > NUMERICAL_CONVERGENCE_THRESHOLD_1)
      {
        keep_going = true;
        break;
      }
    }
    // Modifies the value of the kappas prior to the next iteration, if required.
    if(keep_going)
    {
      it1 = degree_class.begin();
      end = degree_class.end();
      for(; it1!=end; ++it1)
      {
        random_ensemble_kappa_per_degree_class[*it1] += (*it1 - random_ensemble_expected_degree_per_degree_class[*it1]) * uniform_01(engine);
        random_ensemble_kappa_per_degree_class[*it1] = std::fabs(random_ensemble_kappa_per_degree_class[*it1]);
      }
    }
    ++cnt;
  }
  if(cnt >= KAPPA_MAX_NB_ITER_CONV)
  {
    if(!QUIET_MODE) {
      std::clog << std::endl;
      std::clog << TAB << "WARNING: maximum number of iterations reached before convergence. This limit can be"  << std::endl;
      std::clog << TAB << "         adjusted by setting the parameters KAPPA_MAX_NB_ITER_CONV to desired value." << std::endl;
      std::clog << TAB << std::fixed << std::setw(11) << " " << " ";
    }
  }
}


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::infer_parameters(int dim)
{
  if(!QUIET_MODE) { std::clog << "Inferring parameters..."; }

  if (!QUIET_MODE) {
    std::clog << std::endl;
    std::clog << TAB;
    std::clog << std::fixed << std::setw(11) << "beta" << " ";
    std::clog << std::fixed << std::setw(20) << "avg. clustering" << " \n";
  }

  const double BETA_ABS_MIN_DIM = dim + 0.01;
  const double BETA_ABS_MAX_DIM = dim + 100; // what should be the maximum beta?

  if (!CUSTOM_BETA) {
    // Sets initial value to beta, beta > dim
    beta = dim + uniform_01(engine);
    // Iterates until convergence is reached.
    double beta_max = -1;
    double beta_min = dim;
    random_ensemble_average_clustering = 10; 

    while(true) {
      if(!QUIET_MODE) {
        std::clog << TAB;
        std::clog << std::fixed << std::setw(11) << beta << " ";
        std::clog.flush();
      }
      // 1. Infers the values of kappa
      infer_kappas_given_beta_for_degree_class(dim);
      // Computes the cumulative distribution used in the MC integration.
      build_cumul_dist_for_mc_integration(dim);
      // Computes the ensemble clustering.
      compute_random_ensemble_clustering(dim);
      if(!QUIET_MODE) { std::clog << std::fixed << std::setw(20) << random_ensemble_average_clustering << " \n"; }

      // Checks if the expected clustering is close enough. (A.3. last paragraph)
      if (std::fabs(random_ensemble_average_clustering - average_clustering) < NUMERICAL_CONVERGENCE_THRESHOLD_1)
        break;

      // Modifies the bounds on beta if another iteration is required.
      if(random_ensemble_average_clustering > average_clustering)
      {
        beta_max = beta;
        beta = (beta_max + beta_min) / 2;
        if(beta < BETA_ABS_MIN_DIM)
        {
          beta = BETA_ABS_MIN_DIM;
          if(!QUIET_MODE)
            std::clog << "WARNING: value too close to D, using beta = " << std::fixed << std::setw(11) << beta << ".\n";
          break;
        }
      }
      else
      {
        beta_min = beta;
        if(beta_max == -1)
          beta *= 1.5;
        else
          beta = (beta_max + beta_min) / 2;
      }
      if(beta > BETA_ABS_MAX_DIM)
      {
        if(!QUIET_MODE)
          std::clog << "WARNING: value too high, using beta = " << std::fixed << std::setw(11) << beta << ".\n";
        break;
      }
    }
  } else {
    infer_kappas_given_beta_for_degree_class(dim);
    build_cumul_dist_for_mc_integration(dim);
    compute_random_ensemble_clustering(dim);
  }

  // Computes the ensemble average degree.
  compute_random_ensemble_average_degree();
  // Sets the kappas
  kappa.clear();
  kappa.resize(nb_vertices);
  for(int v=0; v < nb_vertices; ++v)
    kappa[v] = random_ensemble_kappa_per_degree_class[degree[v]];

  const auto radius = compute_radius(dim, nb_vertices);
  if(!QUIET_MODE) {
    if(!CUSTOM_BETA)
      std::clog << "                       ";
    std::clog << "...............................................................done."                                         << std::endl;
    std::clog                                                                                                                   << std::endl;
    std::clog << "Inferred ensemble (random positions)"                                                                         << std::endl;
    std::clog << TAB << "Average degree:                 " << random_ensemble_average_degree                                    << std::endl;
    std::clog << TAB << "Minimum degree:                 " << random_ensemble_expected_degree_per_degree_class.begin()->first   << std::endl;
    std::clog << TAB << "Maximum degree:                 " << (--random_ensemble_expected_degree_per_degree_class.end())->first << std::endl;
    std::clog << TAB << "Average clustering:             " << random_ensemble_average_clustering                                << std::endl;
    std::clog << TAB << "Parameters"                                                                                            << std::endl;
    if(!CUSTOM_BETA)
      std::clog << TAB << "  - beta:                       " << beta                                                            << std::endl;
    else
      std::clog << TAB << "  - beta:                       " << beta  << " (custom)"                                            << std::endl;
    std::clog << TAB << "  - mu:                           " << mu                                                              << std::endl;
    std::clog << TAB << "  - radius_S^D (R):               " << radius                              << std::endl;
    std::clog                                                                                                                   << std::endl;
  }

  // Cleans containers that are no longer useful.
  // cumul_prob_kgkp.clear();
  // degree2vertices.clear();
  // random_ensemble_expected_degree_per_degree_class.clear();

  // Initialize random positions
  d_positions.clear();
  d_positions.resize(nb_vertices);
  for (int i=0; i<nb_vertices; ++i)
    d_positions[i] = generate_random_d_vector(dim, radius);
}

// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::infer_parameters()
{
  if(!QUIET_MODE) { std::clog << "Inferring parameters..."; }
  if(!CUSTOM_BETA)
  {
    if (!QUIET_MODE) {
      std::clog << std::endl;
      std::clog << TAB;
      std::clog << std::fixed << std::setw(11) << "beta" << " ";
      std::clog << std::fixed << std::setw(20) << "avg. clustering" << " \n";
    }
    // Sets initial value to beta.
    beta = 2 + uniform_01(engine);
    // Iterates until convergence is reached.
    double beta_max = -1;
    double beta_min = 1;
    random_ensemble_average_clustering = 10;  // dummy value to enter the while loop.
    while( true )
    {
      if(!QUIET_MODE) {
        std::clog << TAB;
        std::clog << std::fixed << std::setw(11) << beta << " ";
        std::clog.flush();
      }
      // Infers the values of kappa.
      infer_kappas_given_beta_for_degree_class();
      // Computes the cumulative distribution used in the MC integration.
      build_cumul_dist_for_mc_integration();
      // Computes the ensemble clustering.
      compute_random_ensemble_clustering();
      if(!QUIET_MODE) { std::clog << std::fixed << std::setw(20) << random_ensemble_average_clustering << " \n"; }

      // Checks if the expected clustering is close enough. (A.3. last paragraph)
      if( std::fabs(random_ensemble_average_clustering - average_clustering) < NUMERICAL_CONVERGENCE_THRESHOLD_1 )
        break;

      // Modifies the bounds on beta if another iteration is required.
      if(random_ensemble_average_clustering > average_clustering)
      {
        beta_max = beta;
        beta = (beta_max + beta_min) / 2;
        if(beta < BETA_ABS_MIN)
        {
          if(!QUIET_MODE)
            std::clog << "WARNING: value too close to 1, using beta = " << std::fixed << std::setw(11) << beta << ".\n";
          break;
        }
      }
      else
      {
        beta_min = beta;
        if(beta_max == -1)
          beta *= 1.5;
        else
          beta = (beta_max + beta_min) / 2;
      }
      if(beta > BETA_ABS_MAX)
      {
        if(!QUIET_MODE)
          std::clog << "WARNING: value too high, using beta = " << std::fixed << std::setw(11) << beta << ".\n";
        break;
      }
    }
  }
  else
  {
    // Infers the values of kappa.
    infer_kappas_given_beta_for_degree_class();
    // Computes the cumulative distribution used in the MC integration.
    build_cumul_dist_for_mc_integration();
    // Computes the ensemble clustering.
    compute_random_ensemble_clustering();
  }

  // Computes the ensemble average degree.
  compute_random_ensemble_average_degree();
  // Sets the kappas.
  kappa.clear();
  kappa.resize(nb_vertices);
  for(int v=0; v<nb_vertices; ++v)
    kappa[v] = random_ensemble_kappa_per_degree_class[degree[v]];

  if(!QUIET_MODE) {
    if(!CUSTOM_BETA)
      std::clog << "                       ";
    std::clog << "...............................................................done."                                         << std::endl;
    std::clog                                                                                                                   << std::endl;
    std::clog << "Inferred ensemble (random positions)"                                                                         << std::endl;
    std::clog << TAB << "Average degree:                 " << random_ensemble_average_degree                                    << std::endl;
    std::clog << TAB << "Minimum degree:                 " << random_ensemble_expected_degree_per_degree_class.begin()->first   << std::endl;
    std::clog << TAB << "Maximum degree:                 " << (--random_ensemble_expected_degree_per_degree_class.end())->first << std::endl;
    std::clog << TAB << "Average clustering:             " << random_ensemble_average_clustering                                << std::endl;
    std::clog << TAB << "Parameters"                                                                                            << std::endl;
    if(!CUSTOM_BETA)
      std::clog << TAB << "  - beta:                       " << beta                                                            << std::endl;
    else
      std::clog << TAB << "  - beta:                       " << beta  << " (custom)"                                            << std::endl;
    std::clog << TAB << "  - mu:                         " << mu                                                                << std::endl;
    std::clog << TAB << "  - radius_S1 (R):              " << nb_vertices / (2 * PI)                                            << std::endl;
    std::clog                                                                                                                   << std::endl;
  }

  // Cleans containers that are no longer useful.
  cumul_prob_kgkp.clear();
  degree2vertices.clear();
  random_ensemble_expected_degree_per_degree_class.clear();

}


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::initialize()
{
  // Sets the default rootname for output files.
  size_t lastdot = EDGELIST_FILENAME.find_last_of(".");
  if(lastdot == std::string::npos)
  {
    ROOTNAME_OUTPUT = EDGELIST_FILENAME;
  }
  ROOTNAME_OUTPUT = EDGELIST_FILENAME.substr(0, lastdot);
  // Initializes the random number generator.
  SEED = std::time(nullptr);
  engine.seed(SEED);
  // Change the stream std::clog to a file.
  if(!QUIET_MODE)
  {
    if(!VERBOSE_MODE)
    {
     logfile.open(ROOTNAME_OUTPUT + ".inf_log");
     // Get the rdbuf of clog.
     // We need it to reset the value before exiting.
     old_rdbuf = std::clog.rdbuf();
     // Set the rdbuf of clog.
     std::clog.rdbuf(logfile.rdbuf());
    }
  }
  // Outputs options and parameters on screen.
  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "===========================================================================================" << std::endl; }
  if(!QUIET_MODE) { std::clog << "Inference of hidden degrees and parameter beta         "                                     << std::endl; }
  if(!QUIET_MODE) { std::clog << "version: "           << VERSION                                                              << std::endl; }
  if(!QUIET_MODE) { std::clog << "started on: "        << format_time(time_started)                                            << std::endl; }
  if(!QUIET_MODE) { std::clog << "edgelist filename: " << EDGELIST_FILENAME                                                    << std::endl; }
  if(!QUIET_MODE) { std::clog << "seed: "              << SEED                                                                 << std::endl; }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "Loading edgelist..."; }
  load_edgelist();
  if(!QUIET_MODE) { std::clog << "...................................................................done."                    << std::endl; }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "Checking number of connected components..."; }
  check_connected_components();
  if(!QUIET_MODE) { std::clog << "............................................done."                                           << std::endl; }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "Analyzing degrees..."; }
  analyze_degrees();
  if(!QUIET_MODE) { std::clog << "..................................................................done."                     << std::endl; }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "Computing local clustering..."; }
  compute_clustering();
  if(!QUIET_MODE) { std::clog << ".........................................................done."                              << std::endl; }

  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }
  if(!QUIET_MODE) { std::clog << "Ordering vertices..."; }
  order_vertices();
  if(!QUIET_MODE) { std::clog << "..................................................................done."                     << std::endl; }
  if(!QUIET_MODE) { std::clog                                                                                                  << std::endl; }

  // Sets the decimal precision of the log.
  std::clog.precision(4);

  // Sets the width of the columns in the output files.
  width_values = 15;
  width_names = 14;
  for(int v(0), l; v<nb_vertices; ++v)
  {
    l = Num2Name[v].length();
    if(l > width_names)
    {
      width_names = l;
    }
  }
  width_names += 1;

  if (!QUIET_MODE) {
      std::clog << "Properties of the graph" << std::endl;
      std::clog << TAB << "Nb vertices:                    " << nb_vertices << std::endl;
      std::clog << TAB << "Nb edges:                       " << nb_edges << std::endl;
      std::clog << TAB << "Average degree:                 " << average_degree << std::endl;
      std::clog << TAB << "Minimum degree:                 " << *(degree_class.begin()) << std::endl;
      std::clog << TAB << "Maximum degree:                 " << *(--degree_class.end()) << std::endl;
      std::clog << TAB << "Nb of degree class:             " << degree_class.size() << std::endl;
      std::clog << TAB << "Average clustering:             " << average_clustering << std::endl;
      std::clog << std::endl;
  }
}

// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::load_edgelist()
{
  // Stream objects.
  std::ifstream edgelist_file;
  std::stringstream one_line;
  // Variables.
  int v1, v2;
  // String objects.
  std::string full_line, name1_str, name2_str;
  // Iterator objects.
  std::map< std::string, int >::iterator name_it;
  // Resets the number of vertices and of edges.
  nb_vertices = 0;
  nb_edges = 0;
  // Resets the containers.
  adjacency_list.clear();
  // Opens the stream and terminates if the operation did not succeed.
  edgelist_file.open(EDGELIST_FILENAME.c_str(), std::ios_base::in);
  if( !edgelist_file.is_open() )
  {
    std::cerr << "Could not open file: " << EDGELIST_FILENAME << "." << std::endl;
    std::terminate();
  }
  else
  {
    // Reads the edgelist file line by line.
    while( !edgelist_file.eof() )
    {
      // Reads a line of the file.
      std::getline(edgelist_file, full_line); edgelist_file >> std::ws;
      one_line.str(full_line); one_line >> std::ws;
      one_line >> name1_str >> std::ws;
      // Skips lines of comment.
      if(name1_str == "#")
      {
        one_line.clear();
        continue;
      }
      one_line >> name2_str >> std::ws;
      one_line.clear();
      // Does not consider self-loops.
      if(name1_str != name2_str)
      {
        // Is name1 new?
        name_it = Name2Num.find(name1_str);
        if( name_it == Name2Num.end() )
        {
          // New vertex.
          v1 = nb_vertices;
          Name2Num[name1_str] = v1;
          Num2Name.push_back(name1_str);
          adjacency_list.emplace_back();
          ++nb_vertices;
        }
        else
        {
          // Known vertex.
          v1 = name_it->second;
        }
        // Is name2 new?
        name_it = Name2Num.find(name2_str);
        if( name_it == Name2Num.end() )
        {
          // New vertex.
          v2 = nb_vertices;
          Name2Num[name2_str] = v2;
          Num2Name.push_back(name2_str);
          adjacency_list.emplace_back();
          ++nb_vertices;
        }
        else
        {
          // Known vertex.
          v2 = name_it->second;
        }
        // Adds the edge to the adjacency list (multiedges are ignored due to std::set).
        std::pair< std::set<int>::iterator, bool > add1 = adjacency_list[v1].insert(v2);
        std::pair< std::set<int>::iterator, bool > add2 = adjacency_list[v2].insert(v1);
        if(add1.second && add2.second) // Both bool should always agree.
        {
          ++nb_edges;
        }
      }
    }
  }
  // Closes the stream.
  edgelist_file.close();
  Name2Num.clear();
}


// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
// =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
void kappas_beta_inference::order_vertices()
{
  // Containers related to the results of the onion decomposition.
  std::vector<int> coreness(nb_vertices);
  std::vector<int> od_layer(nb_vertices);
  // Extracts the onion decomposition.
  extract_onion_decomposition(coreness, od_layer);
  // Orders the vertices based on their layer.
  std::set< std::pair<int, std::pair<double, int> > > layer_set;
  for(int v(0); v<nb_vertices; ++v)
  {
    layer_set.insert(std::make_pair(od_layer[v], std::make_pair(uniform_01(engine), v)));
  }
  // Fills the ordered list of vertices.
  ordered_list_of_vertices.resize(nb_vertices);
  auto it = layer_set.rbegin();
  auto end = layer_set.rend();
  for(int v(0); it!=end; ++it, ++v)
  {
    ordered_list_of_vertices[v] = it->second.second;
  }
  layer_set.clear();
}

int kappas_beta_inference::get_root(int i, std::vector<int> &clust_id)
{
  while(i != clust_id[i])
  {
    clust_id[i] = clust_id[clust_id[i]];
    i = clust_id[i];
  }
  return i;
}

void kappas_beta_inference::merge_clusters(std::vector<int> &size, std::vector<int> &clust_id)
{
  // Variables.
  int v1, v2, v3, v4;
  // Iterators.
  std::set<int>::iterator it, end;
  // Loops over the vertices.
  for(int i(0); i<nb_vertices; ++i)
  {
    // Loops over the neighbors.
    it  = adjacency_list[i].begin();
    end = adjacency_list[i].end();
    for(; it!=end; ++it)
    {
      if(get_root(i, clust_id) != get_root(*it, clust_id))
      {
        // Adjust the root of vertices.
        v1 = i;
        v2 = *it;
        if(size[v2] > size[v1])
          std::swap(v1, v2);
        v3 = get_root(v1, clust_id);
        v4 = get_root(v2, clust_id);
        clust_id[v4] = v3;
        size[v3] += size[v4];
      }
    }
  }
}

void kappas_beta_inference::check_connected_components()
{
  // Vector containing the ID of the component to which each node belongs.
  std::vector<double> Vertex2Prop(nb_vertices, -1);

  // Vector containing the size of the components.
  std::vector<int> connected_components_size;

  // Set ordering the component according to their size.
  std::set< std::pair<int, int> > ordered_connected_components;

  // Starts with every vertex as an isolated cluster.
  std::vector<int> clust_id(nb_vertices);
  std::vector<int> clust_size(nb_vertices, 1);
  for(int v(0); v<nb_vertices; ++v)
  {
    clust_id[v] = v;
  }
  // Merges clusters until the minimal set is obtained.
  merge_clusters(clust_size, clust_id);
  clust_size.clear();
  // Identifies the connected component to which each vertex belongs.
  int nb_conn_comp = 0;
  int comp_id;
  std::map<int, int> CompID;
  for(int v(0); v<nb_vertices; ++v)
  {
    comp_id = get_root(v, clust_id);
    if(CompID.find(comp_id) == CompID.end())
    {
      CompID[comp_id] = nb_conn_comp;
      connected_components_size.push_back(0);
      ++nb_conn_comp;
    }
    Vertex2Prop[v] = CompID[comp_id];
    connected_components_size[CompID[comp_id]] += 1;
  }

  // Orders the size of the components.
  for(int c(0); c<nb_conn_comp; ++c)
  {
    ordered_connected_components.insert( std::make_pair(connected_components_size[c], c) );
  }

  int lcc_id = (--ordered_connected_components.end())->second;
  int lcc_size = (--ordered_connected_components.end())->first;

  if(lcc_size != nb_vertices)
  {
    if(!QUIET_MODE) { std::clog << std::endl; }
    if(!QUIET_MODE) { std::clog << TAB << "- More than one component found!!" << std::endl; }
    if(!QUIET_MODE) { std::clog << TAB << "- " << lcc_size << "/" << nb_vertices << " vertices in the largest component." << std::endl; }
    std::cerr << std::endl;
    std::cerr << "More than one component found (" << lcc_size << "/" << nb_vertices << ") vertices in the largest component." << std::endl;

    std::string edgelist_rootname;
    size_t lastdot = EDGELIST_FILENAME.find_last_of(".");
    if(lastdot == std::string::npos)
    {
      edgelist_rootname = EDGELIST_FILENAME;
    }
    edgelist_rootname = EDGELIST_FILENAME.substr(0, lastdot);

    // Sets the name of the file to write the hidden variables into.
    std::string edgelist_filename = edgelist_rootname + "_GC.edge";
    // Opens the stream and terminates if the operation did not succeed.
    std::fstream edgelist_file(edgelist_filename.c_str(), std::fstream::out);
    if( !edgelist_file.is_open() )
    {
      std::cerr << "Could not open file: " << edgelist_filename << "." << std::endl;
      std::terminate();
    }

    std::set<int>::iterator it, end;
    width_names = 14;
    for(int v1(0), v2, c1, c2; v1<nb_vertices; ++v1)
    {
      c1 = Vertex2Prop[v1];
      if(c1 == lcc_id)
      {
        it  = adjacency_list[v1].begin();
        end = adjacency_list[v1].end();
        for(; it!=end; ++it)
        {
          v2 = *it;
          c2 = Vertex2Prop[v2];
          if(c2 == lcc_id)
          {
            if(v1 < v2)
            {
              edgelist_file << std::setw(width_names) << Num2Name[v1] << " ";
              edgelist_file << std::setw(width_names) << Num2Name[v2] << " ";
              edgelist_file << std::endl;
            }
          }
        }
      }
    }
    // Closes the stream.
    edgelist_file.close();

    if(!QUIET_MODE) { std::clog << TAB << "- Edges belonging to the largest component saved to " << edgelist_rootname + "_GC.edge." << std::endl; }
    if(!QUIET_MODE) { std::clog << TAB << "- Please rerun the program using this new edgelist." << std::endl; }
    if(!QUIET_MODE) { std::clog << std::endl; }
    // if(!QUIET_MODE) { std::clog << "                                          "; }

    if(QUIET_MODE)  { std::clog << std::endl; }
    std::cerr << "Edges belonging to the largest component saved to " << edgelist_rootname + "_GC.edge. Please rerun the program using this new edgelist." << std::endl;
    std::cerr << std::endl;
    // std::terminate();
    std::exit(12); // Custom exist code to rerun Mercator with only GCC
  }
}

inline double kappas_beta_inference::calculateMu() const
{
  return beta * std::sin(PI / beta) / (2.0 * PI * average_degree);
}

inline double kappas_beta_inference::compute_radius(int dim, int N) const
{
  const auto inside = N / (2 * std::pow(PI, (dim + 1) / 2.0)) * std::tgamma((dim + 1) / 2.0);
  return std::pow(inside, 1.0 / dim);
}

inline double kappas_beta_inference::calculate_mu(int dim) const
{
  const auto top = beta * std::tgamma(dim / 2.0) * std::sin(dim * PI / beta);
  const auto bottom = average_degree * 2 * std::pow(PI, 1 + dim / 2.0);
  return top / bottom;
}

std::vector<double> kappas_beta_inference::generate_random_d_vector(int dim, double radius) {
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
    pos = pos / norm * radius;

  return positions;
}

std::vector<double> kappas_beta_inference::generate_random_d_vector_with_first_coordinate(int dim, double angle, double radius) {
  // Assuming that the initial vector v = (1, 0, 0)
  std::vector<double> positions;
  positions.resize(dim + 1);
  double norm{0};
  for (int i = 1; i < dim + 1; i++) { // without the first element
    positions[i] = normal_01(engine);
    norm += positions[i] * positions[i];
  }
  const auto firstNorm = norm / std::sqrt(norm);
  positions[0] = 1.0 / std::tan(angle) * firstNorm;
  normalize_and_rescale_vector(positions, radius);
  return positions;
}

double kappas_beta_inference::compute_angle_d_vectors(const std::vector<double> &v1, const std::vector<double> &v2) {
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

void kappas_beta_inference::normalize_and_rescale_vector(std::vector<double> &v, double radius) {
  int dim = v.size() - 1;
  double norm=0;
  for (int i=0; i<dim + 1; ++i)
    norm += v[i] * v[i];
  
  norm = std::sqrt(norm);
  for (int i=0; i<dim + 1; ++i)
    v[i] /= norm;
  
  for (int i=0; i<dim + 1; ++i)
    v[i] *= radius;
}

void kappas_beta_inference::save_kappas_and_exit() {
  // Save hidden degrees to file
  auto path = std::filesystem::path(EDGELIST_FILENAME);
  path = std::filesystem::absolute(path);
  auto dirname = path.parent_path().string();
  auto filename = path.filename().string();
  std::string kappas_file = filename.substr(0, filename.find("."));
  kappas_file.append(".kappas");
  dirname.append("/");
  dirname.append(kappas_file.c_str()); // dirname + filename.kappas
  std::ofstream outfile(dirname);
  std::for_each(kappa.begin(), kappa.end(), [&outfile](auto x){
    outfile << x << '\n';
  });
  outfile.close();
  std::exit(13);
}

#endif // INFER_KAPPAS_BETA_HPP