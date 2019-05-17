/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 *
 * Modified on : May 16, 2019
 * Author: Ami Woo
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cassert>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std; 

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 1000;    // chosen from experimental observation 
  particles = std::vector<Particle>(static_cast<unsigned long>(num_particles)); // generate particles 
  weights = std::vector<double>(static_cast<unsigned long>(num_particles), 1.0); // deafult weight = 1 
  
   // generate random Gaussian noise 
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  // add random Gaussian noise for each particle
  for (int i = 0; i < num_particles; i++) {
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = weights[i];
    particles[i].id = i; 
  }
  
  is_initialized = true;   //set init flag to true 
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  // generate random Gaussian noise  with mean = 0.0
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) < 0.00001) {   
      // motion model when yaw rate is approximately 0 
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {   
      // motion model when yaw rate ~= 0 
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add random Gaussian noise to each particle 
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  double minDist, curDist ;  // distances to compare 
  
	for (auto &observation : observations) { 
      minDist = numeric_limits<double>::max();
      observation.id = -1;
      
      for (auto const &predObs: predicted) {
      	curDist = dist(predObs.x, predObs.y, observation.x, observation.y);
        if (curDist <= minDist) {
          minDist = curDist;
          observation.id = predObs.id;
        }
      }
    }
     
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  	// variables used for mult-variate Gaussian distribution
  	double norm = 2*M_PI*std_landmark[0]*std_landmark[1];
	double sigmaX = 2*pow(std_landmark[0], 2);
	double sigmaY = 2*pow(std_landmark[1], 2);
  
	for (int i = 0; i < num_particles; i++) {
    	Particle const &particle = particles[i];
      	
      	// transform observation to global coordinates (map coordinates) 
      	vector<LandmarkObs> transformedObs(observations.size());
      	for(int k = 0; k < observations.size(); k++){
        	transformedObs[k].x = particle.x + cos(particle.theta) * observations[k].x - sin(particle.theta) * observations[k].y;
        	transformedObs[k].y = particle.y + sin(particle.theta) * observations[k].x + cos(particle.theta) * observations[k].y;
        	transformedObs[k].id = -1;  //unaware of landmark assocation yet 
        }
      
      	// filter out maps to ensure everything is within the sensor range 
      	vector<LandmarkObs> landmarks;
    	for (auto const &landmark : map_landmarks.landmark_list) {
      		if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
              	LandmarkObs lm = {
                  .id = landmark.id_i,
                  .x = static_cast<double>(landmark.x_f),
                  .y = static_cast<double>(landmark.y_f),
              };
              landmarks.push_back(lm);
            }
        }
      
        if(landmarks.empty()){
            particles[i].weight = 0;
      		weights[i] = 0;
        }
      
      else {
          dataAssociation(landmarks, transformedObs);
          double totalProb = 1.0;
          for (int m = 0; m < transformedObs.size(); m++) {
            auto obs = transformedObs[m];
            
        	auto pred = map_landmarks.landmark_list[obs.id-1];// landmarks[obs.id-1];
            auto dx = obs.x - pred.x_f;
        	auto dy = obs.y - pred.y_f;
        	totalProb *= exp(-(dx * dx / (2 * sigmaX) + dy * dy / (2 * sigmaY))) / norm;
          }
          particles[i].weight = totalProb;
      	  weights[i] = totalProb;
       }
  }
 
}


void ParticleFilter::resample() {
	default_random_engine gen;
  	vector<Particle> resamples(particles.size());
  	discrete_distribution<int> resample_dist(weights.begin(), weights.end());

	for(int i = 0; i<num_particles; i++)
		resamples[i] = particles[resample_dist(gen)];
	
	particles = resamples;
  

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}