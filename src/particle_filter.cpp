/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

const unsigned int NUM_PARTICLES = 10;
const double INITIAL_WEIGHT = 1.0;

#define EPS 0.00001



/****************************************************
* Steps of Particle filter
* 1. Initialiation
* 2. Prediction
* 3. Updation
* 4. Resample
*
*
*****************************************************/



void ParticleFilter::init(double x, double y, double theta, double std[]) {

  this->num_particles = NUM_PARTICLES;


  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < NUM_PARTICLES; i++) {

    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = INITIAL_WEIGHT;

    this->weights.push_back(INITIAL_WEIGHT);
    this->particles.push_back(particle);
  }

  this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	//Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  const bool MOVING_STRAIGHT = fabs(yaw_rate) < EPS;
  
  const double delta_theta = yaw_rate * delta_t;


  default_random_engine gen;
  normal_distribution<double> nx(0.0, std_pos[0]);
  normal_distribution<double> ny(0.0, std_pos[1]);
  normal_distribution<double> ntheta(0.0, std_pos[2]);

  for (int i = 0;  i < NUM_PARTICLES; i++) {

    const double theta = this->particles[i].theta;
    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);
    const double noise_x = nx(gen);
    const double noise_y = ny(gen);
    const double noise_theta = ntheta(gen);

    if (MOVING_STRAIGHT) {

      this->particles[i].x += (velocity * delta_t) * cos_theta + noise_x;
      this->particles[i].y += (velocity * delta_t)* sin_theta + noise_y;
      this->particles[i].theta += noise_theta;

    } else {

      const double phi = theta + delta_theta;
      this->particles[i].x += (velocity / yaw_rate) * (sin(phi) - sin_theta) + noise_x;
      this->particles[i].y += (velocity / yaw_rate) * (cos_theta - cos(phi)) + noise_y;
      this->particles[i].theta = phi + noise_theta;
    }
  }
}



void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	const double BIG_NUMBER = 1.0e99;

  for (int i = 0; i < observations.size(); i++) {

    int current_j;
    double current_smallest_error = BIG_NUMBER;

    for (int j = 0; j < predicted.size(); j++) {

      const double dx = predicted[j].x - observations[i].x;
      const double dy = predicted[j].y - observations[i].y;
      const double error = dx * dx + dy * dy;

      if (error < current_smallest_error) {
        current_j = j;
        current_smallest_error = error;
      }
    }
    observations[i].id = current_j;
  }

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	const double stdx = std_landmark[0];
  const double stdy = std_landmark[1];
  const double na = 0.5 / (stdx * stdx);
  const double nb = 0.5 / (stdy * stdy);
  const double d = sqrt( 2.0 * M_PI * stdx * stdy);

  for (int  i = 0; i < NUM_PARTICLES; i++) {

    const double px = this->particles[i].x;
    const double py = this->particles[i].y;
    const double ptheta = this->particles[i].theta;

    vector<LandmarkObs> landmarks_in_range;
    vector<LandmarkObs> map_observations;

  // Transform observation coordinates.

    for (int j = 0; j < observations.size(); j++){

      const int oid = observations[j].id;
      const double ox = observations[j].x;
      const double oy = observations[j].y;

      const double transformed_x = px + ox * cos(ptheta) - oy * sin(ptheta);
      const double transformed_y = py + oy * cos(ptheta) + ox * sin(ptheta);

      LandmarkObs observation = {
        oid,
        transformed_x,
        transformed_y
      };

      map_observations.push_back(observation);
    }

   // Find map landmarks within the sensor range

    for (int j = 0;  j < map_landmarks.landmark_list.size(); j++) {

      const int mid = map_landmarks.landmark_list[j].id_i;
      const double mx = map_landmarks.landmark_list[j].x_f;
      const double my = map_landmarks.landmark_list[j].y_f;

      const double dx = mx - px;
      const double dy = my - py;
      const double error = sqrt(dx * dx + dy * dy);

      if (error < sensor_range) {

        LandmarkObs landmark_in_range = {
          mid,
          mx,
          my
         };

        landmarks_in_range.push_back(landmark_in_range);
      }
    }

  // Observation association to landmark.
   this->dataAssociation(landmarks_in_range, map_observations);

   // Calculate weights.
    double w = INITIAL_WEIGHT;

    for (int j = 0; j < map_observations.size(); j++){

      const int oid = map_observations[j].id;
      const double ox = map_observations[j].x;
      const double oy = map_observations[j].y;

      const double predicted_x = landmarks_in_range[oid].x;
      const double predicted_y = landmarks_in_range[oid].y;

      const double dx = ox - predicted_x;
      const double dy = oy - predicted_y;

      const double a = na * dx * dx;
      const double b = nb * dy * dy;
      const double r = exp(-(a + b)) / d;
      w *= r;
    }

    this->particles[i].weight = w;
    this->weights[i] = w;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled_particles;

  default_random_engine gen;
  discrete_distribution<int> index(this->weights.begin(), this->weights.end());

  for (int c = 0; c < NUM_PARTICLES; c++) {

    const int i = index(gen);

    Particle p {
      i,
      this->particles[i].x,
      this->particles[i].y,
      this->particles[i].theta,
      INITIAL_WEIGHT
    };

    resampled_particles.push_back(particles[i]);
  }

  this->particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
