int cphoc_f32(const THFloatTensor* X, const THFloatTensor* Y,
              THFloatTensor* R);
int cphoc_f64(const THDoubleTensor* X, const THDoubleTensor* Y,
              THDoubleTensor* R);

int pphoc_f32(const THFloatTensor* X, THFloatTensor* R);
int pphoc_f64(const THDoubleTensor* X, THDoubleTensor* R);


int cphoc_max_f32(const THFloatTensor* X, const THFloatTensor* Y,
                  THFloatTensor* R);
int cphoc_max_f64(const THDoubleTensor* X, const THDoubleTensor* Y,
                  THDoubleTensor* R);

int pphoc_max_f32(const THFloatTensor* X, THFloatTensor* R);
int pphoc_max_f64(const THDoubleTensor* X, THDoubleTensor* R);
