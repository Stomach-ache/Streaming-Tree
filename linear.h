#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H


#ifdef __cplusplus
extern "C" {
#endif

// struct graph_edge
// {
// 	int node1;
// 	int node2;
// 	double weight;
// }

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	double **y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */

	int *numLabels; // multi-label classification
};


enum { L2R_LR, L2R_L2LOSS_SVC, L1R_L2LOSS_SVC, L1R_LR, L2R_L2LOSS_SVC_GD}; /* solver_type */

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double eps2;
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	double p;
	double *init_sol;
  int n_threads; // for parallel
	int init_strat;
	//int all_neg_init; //
	//int mst_schedule;

};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	//double *w;
	struct feature_node **w;
	int *label;		/* label of each class */
	double bias;
};

struct model* train(const struct problem *prob,const struct parameter *param);
// void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, double *target);
// void find_parameter_C(const struct problem *prob, const struct parameter *param, int nr_fold, double start_C, double max_C, double *best_C, double *best_rate);

//double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
//double predict(const struct model *model_, const struct feature_node *x);
//char** predict_all(const struct model *model_, const struct feature_node *x, long long k);
//double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);


//struct model *load_model(const char *model_file_name, struct feature_node **W);
int save_model(const char *model_file_name, const struct model *model_);
//struct model *load_model(const char *model_file_name, struct feature_node **W);
struct model *load_model_stat(const char *model_file_name);
struct feature_node **load_w(const char *model_file_name);
int ** predict(struct feature_node **x, const model *model_, struct feature_node **W, int nr_test, int k, int n_threads);
void evaluate(int ** pred, struct problem * test_prob, int k);


//int save_model(const char *model_file_name, const struct model *model_);
//struct model *load_model(const char *model_file_name, struct feature_node **W);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, long long* label);
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx);
double get_decfun_bias(const struct model *model_, int label_idx);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
//int check_probability_model(const struct model *model);
//int check_regression_model(const struct model *model);
void set_print_string_function(void (*print_func) (const char*));

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */
