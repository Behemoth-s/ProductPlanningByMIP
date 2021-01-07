#include "ortools/linear_solver/linear_solver.h"
namespace operations_research
{
    struct DataModel
    {
        // parameter data
        // demand per product and month
        const std::vector<std::vector<double>> d{
            {400, 400, 800, 800, 1200, 1200, 1200, 1200},
        };
        // setup cost
        const std::vector<double> q{5000};
        // unit cost
        const std::vector<double> p{100};
        // unit stock cost
        const std::vector<double> h{5};
        // intial stock
        const std::vector<double> s_init{200};
        // product types
        const int num_product = 1;
        // months
        const int num_month = 8;
    };

    void MipVarArray()
    {
        DataModel data;
        // create an mip sovler
        MPSolver solver("integer_programming_example",
                        MPSolver::CBC_MIXED_INTEGER_PROGRAMMING);
        // define infinity
        const double infinity = solver.infinity();

        // declare x, s, y
        std::vector<std::vector<const MPVariable *>>
            x(data.num_product, std::vector<const MPVariable *>(data.num_month)),
            s(data.num_product, std::vector<const MPVariable *>(data.num_month)),
            y(data.num_product, std::vector<const MPVariable *>(data.num_month));

        for (int i = 0; i < data.num_product; ++i)
        {
            for (int j = 0; j < data.num_month; ++j)
            {
                x[i][j] = solver.MakeNumVar(0.0, infinity, "");
                s[i][j] = solver.MakeNumVar(0.0, infinity, "");
                y[i][j] = solver.MakeBoolVar("");
            }
        }

        // declare constraint
        for (int i = 0; i < data.num_product; ++i)
        {
            // constraint dem_sat month 1
            // s_init + x[i,1] = d[i, 1] + s[i, 1]
            MPConstraint *dem_sat1 = solver.MakeRowConstraint(
                data.d[i][0] - data.s_init[i], data.d[i][0] - data.s_init[i]);
            dem_sat1->SetCoefficient(x[i][0], 1);
            dem_sat1->SetCoefficient(s[i][0], -1);

            // constraint vub month 1
            // x[i, 1] <= sum(d[i, k] k>=1)*y[i, 1]
            MPConstraint *vub1 = solver.MakeRowConstraint(-infinity, 0);
            //coff_y = sum(d[i, k] k>=1)
            double coff_y = std::accumulate(data.d[i].begin(), data.d[i].end(), 0);
            vub1->SetCoefficient(x[i][0], 1);
            vub1->SetCoefficient(y[i][0], -coff_y);

            for (int j = 1; j < data.num_month; ++j)
            {
                // constraint dem_sat
                // s[i, j-1] + x[i, j] == d[i, j] + s[i, j]
                MPConstraint *dem_sat = solver.MakeRowConstraint(data.d[i][j], data.d[i][j]);
                dem_sat->SetCoefficient(x[i][j], 1);
                dem_sat->SetCoefficient(s[i][j], -1);
                dem_sat->SetCoefficient(s[i][j - 1], 1);

                // constraint vub
                // x[i, j] <= sum(d[i, k] k>=j)*y[i, j]
                MPConstraint *vub = solver.MakeRowConstraint(-infinity, 0);
                //coff_y = sum(d[i, k] k>=j)
                double coff_y = std::accumulate(data.d[i].begin() + j, data.d[i].end(), 0);
                vub->SetCoefficient(x[i][j], 1);
                vub->SetCoefficient(y[i][j], -coff_y);
            }
        }

        // objective cost
        // cost = sum(x[i, j]*q[i]) +sum(y[i, j]*p[i])
        //      + sum(s[i,j]*h[i] j<=8)+ sum(s[i, 8]*h/2)
        MPObjective *const cost = solver.MutableObjective();
        for (int i = 0; i < data.num_product; ++i)
        {
            for (int j = 0; j < data.num_month - 1; ++j)
            {
                cost->SetCoefficient(x[i][j], data.p[i]);
                cost->SetCoefficient(y[i][j], data.q[i]);
                cost->SetCoefficient(s[i][j], data.h[i]);
            }
            cost->SetCoefficient(x[i][data.num_month - 1], data.p[i]);
            cost->SetCoefficient(y[i][data.num_month - 1], data.q[i]);
            cost->SetCoefficient(s[i][data.num_month - 1], data.h[i] / 2);
        }

        cost->SetMinimization();

        const MPSolver::ResultStatus result_status = solver.Solve();

        // Check that the problem has an optimal solution.
        if (result_status != MPSolver::OPTIMAL)
        {
            LOG(FATAL) << "The problem does not have an optimal solution.";
        }
        LOG(INFO) << "Solution:";
        LOG(INFO) << "Optimal objective value = " << cost->Value();
        std::cout << "Optimal x value=";
        for (int i = 0; i < data.num_product; ++i)
        {
            std::cout << "\n";
            for (int j = 0; j < data.num_month; ++j)
            {
                std::cout << x[i][j]->solution_value() << ",";
            }
        }
        std::cout << "\n";
        std::cout << "Optimal y value=";
        for (int i = 0; i < data.num_product; ++i)
        {
            std::cout << "\n";
            for (int j = 0; j < data.num_month; ++j)
            {
                std::cout << y[i][j]->solution_value() << ",";
            }
        }
        std::cout << "\n";
        std::cout << "Optimal s value=";
        for (int i = 0; i < data.num_product; ++i)
        {
            std::cout << "\n";
            for (int j = 0; j < data.num_month; ++j)
            {
                std::cout << s[i][j]->solution_value() << ",";
            }
        }
    }
} // namespace operations_research

int main(int argc, char **argv)
{
    operations_research::MipVarArray();
    return EXIT_SUCCESS;
}