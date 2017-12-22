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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// amount of particles
	num_particles = 100;

    // random number engines
	std::default_random_engine idx;
	std::random_device rd;

	// normal distribution for measurement input
	std::normal_distribution<double> x_(x,std[0]);
	std::normal_distribution<double> y_(x,std[1]);
	std::normal_distribution<double> theta_(x,std[2]);

	// loop through particles to initialise position and noise
	for (unsigned int i=0; i<num_particles; i++){
		// init particle
		Particle p;
		// set particle id
		p.id = i;
		// set weight to 1
		p.weight = 1;
		// get random idx out of normal distribution for each particle
		idx.seed(rd());
		// get random value out of normal distribution
		p.x = x_(idx);
		p.y = y_(idx);
		p.theta = theta_(idx);
		if (p.theta>2*M_PI) p.theta -= 2*M_PI;
		if (p.theta<0) p.theta += 2*M_PI;
		// push particle to particles list
		particles.push_back(p);
		// add initial weight to weight vector
		weights.push_back(1);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// initialise random engine and device
	std::default_random_engine idx;
	std::random_device rd;

	// velocity divided by yaw rate if yaw rate is not zero
	double a;
    
	// loop through particles to add measurements
	for (auto &p:particles){
		// angle prediction step

		if (fabs(yaw_rate)>0.001) {
			double v_yawr = velocity/yaw_rate;
			p.x += v_yawr * (sin(p.theta+yaw_rate*delta_t) - sin(p.theta));
			p.y += v_yawr * (cos(p.theta) - cos(p.theta+yaw_rate*delta_t));
			p.theta += yaw_rate*delta_t;
		} else {
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}
		
		// add gaussian noise
		std::normal_distribution<double> x_(p.x, std_pos[0]);
		std::normal_distribution<double> y_(p.y, std_pos[1]);
		std::normal_distribution<double> theta_(p.theta, std_pos[2]);

		idx.seed(rd());

		p.x = x_(idx);
		p.y = y_(idx);
		p.theta = theta_(idx);
		if (p.theta>2*M_PI) p.theta -= 2*M_PI;
		if (p.theta<0) p.theta += 2*M_PI;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	double error_init = 1.0e99; // big number

	for (int i=0; i<observations.size(); i++){
		double thresh = error_init;
		double new_id = i;
		for (int j=0; j<predicted.size(); j++){
			double delta_x = observations[i].x - predicted[j].x;
			double delta_y = observations[i].y - predicted[j].y;
			double error = sqrt(delta_x * delta_x + delta_y * delta_y);

			if (error<=thresh){
				thresh = error;
				new_id = predicted[j].id;
			}
		}
		observations[i].id = new_id;
	}

	/*
	std::cout<<"Landmarks: "<<std::endl;
	for (auto &pred:predicted){
		std::cout<<"Landmark ID: "<<pred.id<<" x: "<<pred.x<<" y: "<<pred.y<<std::endl;
	}

	std::cout<<"Observations: "<<std::endl;
	for (auto &obs:observations){
		std::cout<<"Observation ID: "<<obs.id<<" x: "<<obs.x<<" y: "<<obs.y<<std::endl;
	}
	*/
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

	double sum_weights = 0.0;
	
	for (auto &p:particles){
		// index for weights vector;
		unsigned int j = 0;
		// vector of transformed observation for each particle
		vector<LandmarkObs> obs_transformed;

		// Transformation of observations
		for (unsigned int i = 0; i<observations.size(); i++){
			LandmarkObs l;
			l.id = observations[i].id;
			l.x = p.x + observations[i].x * cos(p.theta) - observations[i].y * sin(p.theta);
			l.y = p.y + observations[i].y * cos(p.theta) + observations[i].x * sin(p.theta);

			obs_transformed.push_back(l);
		}

		// vector for landmarks in sensor range
		vector<LandmarkObs> lm_in_range;

		// Finding landmarks within sensor range
		for (unsigned int i=0; i<map_landmarks.landmark_list.size(); i++){
			LandmarkObs lm_candidate;

			lm_candidate.id = map_landmarks.landmark_list[i].id_i;
			lm_candidate.x = map_landmarks.landmark_list[i].x_f;
			lm_candidate.y = map_landmarks.landmark_list[i].y_f;

			double delta_x = fabs(p.x - lm_candidate.x);
			double delta_y = fabs(p.y - lm_candidate.y);

			if ( delta_x < sensor_range && delta_y < sensor_range){
				lm_in_range.push_back(lm_candidate);
			}
		}

		// associate landmarks with observations
		dataAssociation(lm_in_range, obs_transformed);

		double w = 1.0; // initial weight
		double scaling = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
		double sigma_x2 = 2*std_landmark[0]*std_landmark[0];
		double sigma_y2 = 2*std_landmark[1]*std_landmark[1];

		for (auto &obst:obs_transformed){
			for (auto &lm:lm_in_range){

				//std::cout<<"Observed coordinates: "<<pred_x<<" "<<pred_y<<std::endl;
				//std::cout<<"Associated coordinates: "<<obst.x<<" "<<obst.y<<std::endl;
				if (lm.id==obst.id){
					double delta_x = obst.x - lm.x;
					double delta_x2 = delta_x*delta_x;
					double delta_y = obst.y - lm.y;
					double delta_y2 = delta_y*delta_y;

					w *= scaling * exp( -( (delta_x2 / sigma_x2) + (delta_y2 / sigma_y2) ) );
					//std::cout<<"Landmark coordinates: "<<lm.x<<" "<<lm.y<<" observation coordinates "<<obst.x<<" "<<obst.y<<" new weight: "<<w<<std::endl;
				}
			}
		}
		p.weight = w;
		weights[j] = w;
		j++;
		sum_weights += w;
		//std::cout<< "Weight of Particle number "<< p.id <<" is "<<p.weight<<std::endl;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> new_particles(num_particles);
	
	std::random_device seed;
	std::default_random_engine random_generator(seed());
	std::discrete_distribution<int> index(weights.begin(), weights.end());

	for (int i = 0; i<num_particles; i++){
		const int idx = index(random_generator);

		new_particles[i] = particles[idx];
	}

	particles.clear();
	particles = new_particles;
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
