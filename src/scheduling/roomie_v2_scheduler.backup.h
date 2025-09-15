#ifndef ROOMIE_V2_SCHEDULER_H
#define ROOMIE_V2_SCHEDULER_H

#include <math.h>
#include <random>
#include <algorithm>
#include "utils/general.h"
#include "utils/datastore.h"
#include "roomie_v1_scheduler.h"

class RoomieV2Scheduler : public RoomieV1Scheduler
{
public:
  RoomieV2Scheduler() {}

  std::pair<std::vector<double>, std::vector<double>> heuristic_roomie(std::vector<Model *> &models, float prob = 0.5)
  {
    std::vector<double> durations, new_durations;
    std::vector<int> lengths;
    std::vector<std::vector<std::vector<double>>> masks;

    int N = models.size();

    if (N == 1)
    {
      auto duration = models[0]->initial_duration();
      durations.push_back(duration);
      return {durations, durations};
    }

    for (int i = 0; i < N; ++i)
    {
      auto duration = models[i]->initial_duration();
      durations.push_back(duration);
      new_durations.push_back(duration);
      lengths.push_back(models[i]->get_kernels().size());
      std::vector<double> kernel_durations;
      std::vector<float> occ;
      for (auto &kernel : models[i]->get_kernels())
      {
        kernel_durations.push_back(kernel->duration);
      }
      masks.push_back(create_mask(kernel_durations));
    }

    for (int i = 0; i < N; ++i)
    {
      for (int j = 0; j < N; ++j) /* model to interfere with */
      {
        if (i == j)
        {
          continue;
        }

        auto mask_durations = masks[j];

        std::vector<double> delays;
        for (const auto &sample_dur : mask_durations)
        {
          double delayed = 0.0;
          for (size_t k = 0; k < sample_dur.size(); k++)
          {
            /* Early stopping */
            if (k > masks[i].size())
            {
              break;
            }
            if (bernoulli(prob))
            {
              delayed += sample_dur[k];
            }
          }
          delays.push_back(delayed);
        }
        /* Determine the number of time the model(j) might terminate while model(j) still executing. */
        auto overlap = std::max((float)lengths[i] / lengths[j] / 2, 1.0f);
        new_durations[i] += overlap * median(delays);
      }
    }

    return {durations, new_durations};
  }
};

#endif // ROOMIE_V2_SCHEDULER_H