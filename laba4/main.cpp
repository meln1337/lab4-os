#include <iostream>
#include <thread>
#include <vector>
#include <fstream>
#include <chrono>
//#include <omp.h>
#include <string>
#include <math.h>
//#include <future>
//#include <Windows.h>

#define MAX_THREADS 1000

typedef long long int ll;

void generate_file(ll n, ll m, ll d, ll e)
{
	std::ofstream file(std::to_string(n) + "x" + std::to_string(m) + "_" + std::to_string(d) + "x" + std::to_string(e) + ".txt");
	file << n << " " << m << std::endl;
	ll k = 0;
	for (ll i = 0; i < n; i++)
	{
		for (ll j = 0; j < m; j++)
		{
			k++;
			if (j == m - 1)
			{
				file << k;
			}
			else
			{
				file << k << " ";
			}
		}
		file << std::endl;
	}
	file << d << " " << e;
	k = 0;
	for (ll i = 0; i < d; i++)
	{
		for (ll j = 0; j < e; j++)
		{
			k++;
			if (j == e - 1)
			{
				file << k;
			}
			else
			{
				file << k << " ";
			}
		}
		file << std::endl;
	}
	file.close();
}

std::vector<std::vector<ll>> matrix_multiplication(std::vector<std::vector<ll>> a, std::vector<std::vector<ll>> b)
{
	// a (n, m), b (d, e) => res (n, e)
	auto start = std::chrono::high_resolution_clock::now();
	ll n = a.size();
	ll m = a[0].size();
	ll e = b[0].size();
	std::vector<std::vector<ll>> res(n, std::vector<ll>(e, 0));
	for (ll i = 0; i < n; i++)
	{
		for (ll j = 0; j < e; j++)
		{
			for (ll k = 0; k < m; k++)
			{
				res[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Time taken by function matrix_multiplication: " << duration.count() << " milliseconds" << std::endl;
	return res;
}

void vector_multiplication(std::vector<std::vector<ll>> a, std::vector<std::vector<ll>> b, std::vector<std::vector<ll>>& c, std::vector<std::pair<int, int>> indexes, std::vector<bool>& done, int i)
{
	//std::cout << "thread id: " << std::this_thread::get_id() << std::endl;
	for (int idx = 0; idx < indexes.size(); idx++)
	{
		for (int k = 0; k < b.size(); k++)
		{
			int x = indexes[idx].first;
			int y = indexes[idx].second;
			c[x][y] += a[x][k] * b[k][y];
		}
	}
	done[i] = true;
}

int get_status(std::vector<bool> done)
{
	int count = 0;
	for (int i = 0; i < done.size(); i++) if (done[i]) count++;
	return count;
}

void progress_bar(double progress)
{	
	int barWidth = 70;

	std::cout << "[";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progress * 100.0) << " %\r";
	std::cout.flush();

	std::cout << std::endl;
}

void print_matrix_to_file(std::vector<std::vector<ll>> a, std::string name);

std::vector<std::vector<ll>> matrix_multiplication_parallel(std::vector<std::vector<ll>> a, std::vector<std::vector<ll>> b, int NUM_THREADS, int NUM_CONCURRENT, std::vector<std::vector<std::pair<int, int>>> indexes, std::string filename)
{
	// a (n, m), b (d, e) => res (n, e)
	auto start = std::chrono::high_resolution_clock::now();
	ll n = a.size();
	ll m = a[0].size();
	ll e = b[0].size();
	if (NUM_THREADS > n * e)
	{
		std::cout << "Too many threads" << std::endl;
		NUM_THREADS = n * e;
	}
	if (NUM_CONCURRENT > NUM_THREADS)
	{
		NUM_CONCURRENT = NUM_THREADS;
	}
	std::vector<std::thread> th(NUM_THREADS);
	std::vector<std::vector<ll>> c(n, std::vector<ll>(e, 0));

	/*for (lli i = 0; i < n; i++)
	{
		for (lli j = 0; j < e; j++)
		{
			for (lli k = 0; k < m; k++)
			{
				res[i][j] += a[i][k] * b[k][j];
			}
		}
	}*/
	int NUM_BATCHES = ceil(double(NUM_THREADS) / NUM_CONCURRENT);
	std::vector<bool> done(NUM_THREADS, false);
	int status = 0;
	int seconds = 0;
	if (NUM_BATCHES == 1)
	{
		for (int i = 0; i < NUM_THREADS; i++) th[i] = std::thread(vector_multiplication, a, b, ref(c), indexes[i], ref(done), i);
		
		while (status != NUM_BATCHES)
		{
			status = get_status(done);
			progress_bar((double)status / NUM_THREADS);
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			seconds++;
			print_matrix_to_file(c, filename + "_" + std::to_string(seconds) + "s.txt");
		}

		for (int i = 0; i < NUM_THREADS; i++) th[i].join();
	}
	else
	{
		for (int i = 0; i < NUM_BATCHES; i++)
		{
			if (i == NUM_BATCHES - 1)
			{
				for (int j = 0; j < NUM_THREADS - i * NUM_CONCURRENT; j++)
				{
					th[i * NUM_CONCURRENT + j] = std::thread(vector_multiplication, a, b, ref(c), indexes[i * NUM_CONCURRENT + j], ref(done), i * NUM_CONCURRENT + j);
				}

				while (status != NUM_THREADS)
				{
					status = get_status(done);
					progress_bar((double)status / NUM_THREADS);
					std::this_thread::sleep_for(std::chrono::milliseconds(1000));
					seconds++;
					print_matrix_to_file(c, filename + "_" + std::to_string(seconds) + "s.txt");
				}

				for (int j = 0; j < NUM_THREADS - i * NUM_CONCURRENT; j++)
				{
					th[i * NUM_CONCURRENT + j].join();
				}
			}
			else
			{
				for (int j = 0; j < NUM_CONCURRENT; j++)
				{
					th[i * NUM_CONCURRENT + j] = std::thread(vector_multiplication, a, b, ref(c), indexes[i * NUM_CONCURRENT + j], ref(done), i * NUM_CONCURRENT + j);
				}
				
				while (status != (i+1) * NUM_CONCURRENT)
				{
					status = get_status(done);
					progress_bar((double)status / NUM_THREADS);
					std::this_thread::sleep_for(std::chrono::milliseconds(1000));
					seconds++;
					print_matrix_to_file(c, filename + "_" + std::to_string(seconds) + "s.txt");
				}

				for (int j = 0; j < NUM_CONCURRENT; j++)
				{
					th[i * NUM_CONCURRENT + j].join();
				}
			}
			std::cout << std::endl;
		}
	}
	
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Time taken by function: matrix_multiplication_parallel " << duration.count() << " milliseconds" << std::endl;
	return c;
}

void print_matrix(std::vector<std::vector<ll>> a)
{
	for (ll i = 0; i < a.size(); i++)
	{
		for (ll j = 0; j < a[0].size(); j++) std::cout << a[i][j] << " ";
		std::cout << std::endl;
	}
}

void print_matrix_to_file(std::vector<std::vector<ll>> a, std::string name)
{
	std::ofstream file(name);
	for (ll i = 0; i < a.size(); i++)
	{
		for (ll j = 0; j < a[0].size(); j++) file << a[i][j] << " ";
		file << std::endl;
	}
	file.close();
}

int main()
{
	//generate_file(200, 400, 400, 800);
	std::string name = "D:\\univer\\3 семестр\\ос\\laba4\\laba4\\100x200_200x300";
	int NUM_THREADS, NUM_CONCURRENT;
	std::cout << "Enter NUM_THREADS = ";
	std::cin >> NUM_THREADS;
	std::cout << "Enter NUM_CONCURRENT = ";
	std::cin >> NUM_CONCURRENT;
	std::ifstream file(name + ".txt");
	ll n, m;
	file >> n >> m;
	std::vector<std::vector<ll>> a(n, std::vector<ll>(m));
	for (ll i = 0; i < n; i++)
	{
		for (ll j = 0; j < m; j++) file >> a[i][j];
	}
	ll d, e;
	file >> d >> e;
	std::vector<std::vector<ll>> b(d, std::vector<ll>(e));
	for (ll i = 0; i < d; i++)
	{
		for (ll j = 0; j < e; j++) file >> b[i][j];
	}
	if (NUM_THREADS > n * e)
	{
		std::cout << "Too many threads" << std::endl;
		NUM_THREADS = n * e;
	}
	std::vector < std::vector < std::pair<int, int >> > indexes(NUM_THREADS, std::vector<std::pair<int, int>>());
	int count = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < e; j++)
		{
			indexes[count % NUM_THREADS].push_back({ i, j });
			count++;
		}
	}
	/*for (int i = 0; i < NUM_THREADS; i++)
	{
		cout << i << "-th thread: ";
		for (int j = 0; j < indexes[i].size(); j++)
		{
			cout << "{" << indexes[i][j].first << ", " << indexes[i][j].second << "}" << ", ";
		}
		cout << endl;
	}*/
	/*std::vector<std::vector<ll>> c1 = matrix_multiplication(a, b);
	print_matrix_to_file(c1, name + "_output.txt");*/
	std::vector<std::vector<ll>> c2 = matrix_multiplication_parallel(a, b, NUM_THREADS, NUM_CONCURRENT, indexes, name);
	print_matrix_to_file(c2, name + "output_parallel.txt");
	//this_thread::sleep_until(chrono::high_resolution_clock::now() + chrono::seconds(1));
	//print_matrix(c1);
	//cout << endl;
	//print_matrix(c2);
	file.close();
	return 0;
}