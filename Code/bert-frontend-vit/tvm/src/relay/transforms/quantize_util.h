#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <cmath>
#include <cfloat>
#include <algorithm>

#define FL_SEARCH_MAX 32 // 25

/*
 * Quantize data to wordlen-bit fix-point format with fraclen-bit for fraction
 * use_round indicates round/truncation
 * return quantization error corresponding to scheme (wordlen, fraclen, use_round)
 *
 * data_quantized = round(clip_by_max_min(data/step))*step
 */
template<typename T> 
float Convert(T* data, size_t size, int wordlen, int fraclen,
    bool use_round, float* data_o) {
  // Precision
  float step = std::pow(2, -fraclen);
  // Bound
  float step_upper_bound = std::pow(2, wordlen-1)-1;
  float step_lower_bound = -std::pow(2, wordlen-1);
  float q_error = 0;
  size_t non_zero_cnt = 0;
  for (size_t i=0; i < size; i++) {
    float n_step = data[i]/step;
    // Round/truncation
    if (use_round) {
      n_step = std::round(n_step);
    } else {
      n_step = std::trunc(n_step);
    }
    // Deal with overflow
    if (n_step > step_upper_bound) {
      n_step = step_upper_bound;
    } else if (n_step < step_lower_bound) {
      n_step = step_lower_bound;
    }
    // Get quantized value
    data_o[i] = n_step*step;
    // Keep record of quantization error
    q_error += (data[i] - data_o[i]) * (data[i] - data_o[i]);
    if (data[i] != data_o[i]) {
      non_zero_cnt += 1;
    }
  }
  return q_error;
}

template<typename T>
int SearchFraclen(T* dl, int size, int wordlen) {
  // Find fraction length for max and min
  auto max_iter = std::max_element(dl, dl+size);
  auto max_index = std::distance(dl, max_iter);
  T max = dl[max_index];
  int fl_max = wordlen - 1 - std::lround(std::log2(std::abs(max)));
  auto min_iter = std::min_element(dl, dl+size);
  auto min_index = std::distance(dl, min_iter);
  T min = dl[min_index];
  int fl_min = wordlen - 1 - std::lround(std::log2(std::abs(min)));
  // Search for best fraction length
  float error = FLT_MAX;
  int fl = 0;
  std::vector<float> errors;
  float* tmp = new float[size];
  // Check the quantization error with fraction length i std::max(fl_max, fl_min)
  for (int i = std::min(FL_SEARCH_MAX - 1, std::min(fl_max, fl_min) - 2);
           i < FL_SEARCH_MAX; i++) {
    float e = Convert<T>(dl, size, wordlen, i, true, tmp);
    if (e < error) {
      error = e;
      fl = i;
    }
    if (errors.size() > 0 && e > errors.back()) {
      break;
    }
    errors.push_back(e);
  }
  delete tmp;
  return fl;
}
