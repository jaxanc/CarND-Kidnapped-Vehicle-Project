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

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	num_particles = 10;	 

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (unsigned int i = 0; i != num_particles; ++i)
	{
		Particle currentParticle;
		currentParticle.id = i;
		currentParticle.x = dist_x(gen);
		currentParticle.y = dist_y(gen);
		currentParticle.theta = dist_theta(gen);
		currentParticle.weight = 1.0;
		particles.push_back(currentParticle);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	double dx = 0;
	double dy = 0;
	double dtheta = 0;

	for (unsigned int i = 0; i != num_particles; ++i)
	{
		Particle &currentParticle = particles[i];

		if (fabs(yaw_rate) > 0.0001)
		{
			dx = velocity/yaw_rate*(sin(currentParticle.theta+yaw_rate*delta_t)-sin(currentParticle.theta));
			dy = velocity/yaw_rate*(cos(currentParticle.theta)-cos(currentParticle.theta+yaw_rate*delta_t));
			dtheta = yaw_rate*delta_t;	
		}
		else
		{
			dx = velocity*cos(currentParticle.theta)*delta_t;
			dy = velocity*sin(currentParticle.theta)*delta_t;
			dtheta = 0;
		}
		
		normal_distribution<double> dist_x(dx, std_pos[0]);
		normal_distribution<double> dist_y(dy, std_pos[1]);
		normal_distribution<double> dist_theta(dtheta, std_pos[2]);

		currentParticle.x += dist_x(gen);
		currentParticle.y += dist_y(gen);
		currentParticle.theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	for (unsigned int i = 0; i < observations.size(); ++i)
	{
		LandmarkObs &currentObservation = observations[i];

		double minDistance = 9999999;
		int minDistance_idx = 0;

		for (unsigned int j = 0; j < predicted.size(); ++j)
		{
			LandmarkObs &currentPredicted = predicted[j];

			double currentDistance = dist(currentObservation.x, currentObservation.y, currentPredicted.x, currentPredicted.y);
			if (currentDistance < minDistance)
			{
				minDistance = currentDistance;
				minDistance_idx = j;
			}
		}

		currentObservation.id = minDistance_idx;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks)
{
	for (unsigned int i = 0; i < num_particles; ++i)
	{
		Particle &currentParticle = particles[i];
		std::vector<LandmarkObs> predicted;
		std::vector<LandmarkObs> observationsT = observations;

		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j)
		{
			LandmarkObs predictedLandMark;
			predictedLandMark.x = map_landmarks.landmark_list[j].x_f;
			predictedLandMark.y = map_landmarks.landmark_list[j].y_f;
			predictedLandMark.id = map_landmarks.landmark_list[j].id_i;

			predicted.push_back(predictedLandMark);
		}

		for (unsigned int j = 0; j< observationsT.size(); ++j)
		{
		  
		  observationsT[j].x = currentParticle.x + (cos(currentParticle.theta)*observations[j].x) - (sin(currentParticle.theta) * observations[j].y);
		  observationsT[j].y = currentParticle.y + (sin(currentParticle.theta)*observations[j].x) + (cos(currentParticle.theta) * observations[j].y);
		}

		dataAssociation(predicted, observationsT);

		double weight = 1.0;
		for (unsigned int j = 0; j< observationsT.size(); ++j)
		{
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];
			double x_obs = observationsT[j].x;
			double y_obs = observationsT[j].y;
			double mu_x = map_landmarks.landmark_list[observationsT[j].id].x_f;
			double mu_y = map_landmarks.landmark_list[observationsT[j].id].y_f;

			// calculate normalization term
			double gauss_norm = (1/(2 * M_PI * sig_x * sig_y));

			// calculate exponent
			double exponent = pow(x_obs - mu_x, 2)/(2*pow(sig_x, 2))
							+ pow(y_obs - mu_y, 2)/(2*pow(sig_y, 2));

			// calculate weight using normalization terms and exponent
			weight *= gauss_norm * exp(-exponent);
		}

		currentParticle.weight = weight;
		weights[i] = weight;
	}
}

void ParticleFilter::resample()
{
	std::vector<Particle> newParticles;

	std::discrete_distribution<int> weightsDistribution (weights.begin(), weights.end());

	for (unsigned int i = 0; i < num_particles; ++i)
	{
		newParticles.push_back(particles[weightsDistribution(gen)]);
	}

	particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
