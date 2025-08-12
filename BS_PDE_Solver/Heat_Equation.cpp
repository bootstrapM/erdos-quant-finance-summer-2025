#include<iostream>
#include<cmath>
#include<vector>
#include<fstream>
#include<stdexcept>

std::vector<double> tridiagonal_solver(std::vector<double> &a,
    std::vector<double> &b,
    std::vector<double> &c,
    std::vector<double> &d);

void print_vector(std::vector<double> &v) 
{
    for(auto el: v)
    {
        std::cout<<el<<" ";
    }
    std::cout<<std::endl;
}

void checking_tridiagonal_solver()
{
    std::vector<double> a{-1, -1, -1, -1, -1};
    std::vector<double> c{-1, -1, -1, -1, -1};
    std::vector<double> b{2, 2, 2, 2, 2, 2};
    // std::vector<double> d{3, 4, 2, 3, 9, 1};
    std::vector<double> d{1,2,3,4,5,6};

    std::vector<double> ans = tridiagonal_solver(a, b, c, d);

    for(auto v: ans)
    {
        std::cout<<v<<" ";
    }
}


std::vector<double> tridiagonal_solver(std::vector<double> &a,
                        std::vector<double> &b,
                        std::vector<double> &c,
                        std::vector<double> &d)
{
    // this function implements a solver for tridiagonal linear system of equations
    int m = b.size();
    if(a.size()!=m-1 || c.size()!=m-1 || d.size()!= m)
    {
        throw std::invalid_argument("Sizes of vectors constituting the tri-diagonal matrix are inconsistent");
    }

    // Converting the tri-diagonal matrix into upper diagonal form
    for(int i=1; i<m; i++)
    {
        double weight = a[i-1] / b[i-1];
        b[i] -= weight * c[i-1];
        d[i] -= weight * d[i-1];
    }

    // debug:
    // std::cout<<"b vector is"<<std::endl;
    // print_vector(b);
    // std::cout<<"d vector is"<<std::endl;
    // print_vector(d);

    // now back substitute from the reverse
    std::vector<double> v(m);
    v[m-1] = d[m-1]/b[m-1];
    for(int i=m-2; i>=0; i--)
    {
        v[i] = (d[i] - c[i]* v[i+1]) / b[i];
    }
    return v;
}

void CN_Solver()
{
    const double pi = 3.14159; 
    // diffusion constant
    const double D = 0.5;

    // x_min and x_max positional boundaries
    const double x_min = 0.0; 
    const double x_max = 1.0;

    // max time to evolve to
    const double T = 1.0;

    // specify number of steps for the numerical simulation
    const int num_t_steps = 5000; 
    const int num_x_steps = 500;

    // specify the grid size
    const double x_step_size = (x_max-x_min) / static_cast<double>(num_x_steps);
    const double t_step_size = T / static_cast<double>(num_t_steps);

    const double alpha  = -D * t_step_size / (2* x_step_size * x_step_size);

    // layout the spatial grid
    std::vector<double> x_grid(num_x_steps+1); // vector x_grid contains the location of the grid points
    for(int i=0; i<=num_x_steps; i++)
    {
        x_grid[i] = i * x_step_size; 
    }

    // initial condition: u(x,0) = g(x) = sin(pi * x / L)
    std::vector<double> g(num_x_steps+1);
    for(int i=0; i<=num_x_steps; i++)
    {
        g[i] = std::sin(pi * x_grid[i]); 
    }

    // solution vector at time step n
    std::vector<double> u_n(num_x_steps+1);
    // solution vector at time step n+1
    // std::vector<double> u_new(num_x_steps+1);

    u_n = g;

    // to get u_new we need to first setup the tridiagonal matrix and solve the system
    // α u_{i+1, n+1} + (1-2α) u_{i,n+1} + α u_{i-1, n+1} = -α u_{i+1, n} + (1+2α) u_{i,n} - α u_{i-1, n}
    std::vector<double> a_vec(num_x_steps, alpha);
    std::vector<double> b_vec(num_x_steps+1, 1-2*alpha);
    std::vector<double> b_vec_temp(num_x_steps+1, 1-2*alpha);
    std::vector<double> c_vec(num_x_steps, alpha);
    std::vector<double> d_vec(num_x_steps+1);

    c_vec[0] = 0.0, a_vec[num_x_steps-1] = 0.0, b_vec[0] = 1, b_vec[num_x_steps] = 1;

    for(int n=0; n<num_t_steps; n++)
    {
        b_vec_temp = b_vec;
        for(int i=1; i<=num_x_steps-1; i++)
        {
            d_vec[i] = - alpha * u_n[i+1] + (1+2 * alpha) * u_n[i] - alpha * u_n[i-1];
        }

        // Dirichlet boundary conditions: u(0,t)=0, u(1,t)=0
        d_vec[0] = 0.0;
        d_vec[num_x_steps] = 0.0;
        
        u_n = tridiagonal_solver(a_vec, b_vec_temp, c_vec, d_vec);
    }

    std::ofstream fout("heat_eq_sol.txt");
    fout<<"x"<<" "<<"u(x)"<<std::endl;
    for(int i=0; i<=num_x_steps; i++ )
    {
        fout<<x_grid[i]<<" "<<u_n[i]<<std::endl;
    }
    fout.close();
    std::cout<<"Wrote output to file";

}


int main()
{
    // checking_tridiagonal_solver();
    CN_Solver();
    return 0;
}