#include <core.p4>
#include <v1model.p4>
typedef bit<32> STATE;
typedef bit<32> ACTION;









typedef struct qtable {
    double **table;
    STATE state_number;
    ACTION action_number;
} QTable;

QTable *init_qtable(STATE state_number, ACTION action_number);
void destroy_qtable(QTable *qtable);

double Q(const QTable *qtable, STATE s, ACTION a);
double max_Q(const QTable *qtable, STATE s);

void save_qtable(QTable *qtable, const char *filePath);
QTable *load_qtable(const char *filePath);