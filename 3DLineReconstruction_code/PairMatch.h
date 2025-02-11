#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include"SfMManager.h"
#include"MatchManager.h"

#include <mutex>
#include <map>
#include <unordered_map>
#include <queue>
#include <condition_variable>
#include <memory>
#include <thread>
#pragma region ThreadSafeQuene
template<typename T>
class Threadsafe_Queue
{
private:
	mutable std::mutex mut;
	//std::queue<T> data_queue;
	std::queue<std::shared_ptr<T>> data_queue;
	std::condition_variable data_cond;

public:
	Threadsafe_Queue() {}
	Threadsafe_Queue(Threadsafe_Queue const& other)
	{
		std::lock_guard<std::mutex> lk(other.mut);
		data_queue = std::move(other.data_queue);
	}

	void push(T new_value)
	{
		std::shared_ptr<T> data(std::make_shared<T>(std::move(new_value)));
		std::lock_guard<std::mutex> lk(mut);
		data_queue.push(data);
		data_cond.notify_one();
	}
	//void push(T new_value)
	//{
	//	std::lock_guard<std::mutex> lk(mut);
	//	data_queue.push(std::move(new_value));
	//	data_cond.notify_one();
	//}

	//void wait_and_pop(T& value)
	//{
	//	std::unique_lock<std::mutex> lk(mut);
	//	data_cond.wait(lk, [this] {return !data_queue.empty(); });
	//	value = std::move(data_queue.front());
	//	data_queue.pop();
	//}
	void wait_and_pop(T& value)
	{
		std::unique_lock<std::mutex> lk(mut);
		data_cond.wait(lk, [this] {return !data_queue.empty(); });
		value = std::move(*data_queue.front());
		data_queue.pop();
	}

	std::shared_ptr<T> wait_and_pop()
	{
		std::unique_lock<std::mutex> lk(mut);
		data_cond.wait(lk, [this] {return !data_queue.empty(); });
		std::shared_ptr<T> res = data_queue.front();
		//std::shared_ptr<T> res(std::make_shared<T>(std::move(data_queue.front())));
		data_queue.pop();
		return res;
	}

	bool try_pop(T& value)
	{
		std::lock_guard<std::mutex> lk(mut);
		if (data_queue.empty())
			return false;
		value = std::move(*data_queue.front());
		data_queue.pop();
		return true;
	}

	std::shared_ptr<T> try_pop()
	{
		std::lock_guard<std::mutex> lk(mut);
		if (data_queue.empty())
			return std::shared_ptr<T>();
		std::shared_ptr<T> res = data_queue.front();
		//std::shared_ptr<T> res(std::make_shared<T>(std::move(data_queue.front())));
		data_queue.pop();
		return res;
	}

	bool empty() const
	{
		std::lock_guard<std::mutex> lk(mut);
		return data_queue.empty();
	}

	int size() const
	{
		std::lock_guard<std::mutex> lk(mut);
		return data_queue.size();
	}

};

template<typename T>
class Threadsafe_Queue_with_Node
{
private:
	struct node
	{
		std::shared_ptr<T> data;
		std::unique_ptr<node> next;
	};

	std::mutex head_mutex;
	std::unique_ptr<node> head;
	std::mutex tail_mutex;
	node* tail;
	std::condition_variable data_cond;

	node* get_tail()
	{
		std::lock_guard<std::mutex> tail_lock(tail_mutex);
		return tail;
	}

	std::unique_ptr<node> pop_head()
	{
		//std::lock_guard<std::mutex> head_lock(head_mutex);
		std::unique_ptr<node> old_head = std::move(head);
		//if (head.get()==get_tail())
		//{
		//	return nullptr;
		//}
		//std::unique_ptr<node> old_head = std::move(head);
		head = std::move(old_head->next);
		return old_head;
	}

	std::unique_lock<std::mutex> wait_for_data()
	{
		std::unique_lock<std::mutex> head_lock(head_mutex);
		data_cond.wait(head_lock, [&] {return head.get() != get_tail(); });
		return std::move(head_lock);
	}

	std::unique_ptr<node> wait_pop_head()
	{
		std::unique_lock<std::mutex> head_lock(wait_for_data());
		return pop_head();
	}

	std::unique_ptr<node> wait_pop_head(T& value)
	{
		std::unique_lock<std::mutex> head_lock(wait_for_data());
		value = std::move(*head->data);
		return pop_head();
	}

	std::unique_ptr<node> try_pop_head()
	{
		std::lock_guard<std::mutex> head_lock(head_mutex);
		if (head.get() == get_tail())
		{
			return std::unique_ptr<node>();
		}
		return pop_head();
	}

	std::unique_ptr<node> try_pop_head(T& value)
	{
		std::lock_guard<std::mutex> head_lock(head_mutex);
		if (head.get() == get_tail())
		{
			return std::unique_ptr<node>();
		}
		value = std::move(*head->data);
		return pop_head();
	}

public:
	Threadsafe_Queue_with_Node() :head(new node), tail(head.get())
	{}
	Threadsafe_Queue_with_Node(const Threadsafe_Queue_with_Node& other) = delete;
	Threadsafe_Queue_with_Node& operator=(const Threadsafe_Queue_with_Node& ohter) = delete;

	std::shared_ptr<T> try_pop()
	{
		//std::unique<T> old_head = pop_head();
		//return old_head ? old_head->data : std::shared_ptr<T>();
		std::unique_ptr<node> old_head = try_pop_head();
		return old_head ? old_head->data : std::shared_ptr<T>();
	}

	bool try_pop(T& value)
	{
		std::unique_ptr<node> const old_head = try_pop_head(value);
		return old_head != nullptr;
	}

	std::shared_ptr<T> wait_and_pop()
	{
		std::unique_ptr<node> const old_head = wait_pop_head();
		return old_head->data;
	}

	void wait_and_pop(T& value)
	{
		std::unique_ptr<node> const old_head = wait_pop_head(value);
	}

	void push(T new_value)
	{
		std::shared_ptr<T> new_data(std::make_shared<T>(std::move(new_value)));
		std::unique_ptr<node> p(new node);
		{
			std::lock_guard<std::mutex> tail_lock(tail_mutex);
			tail->data = new_data;
			node* const new_tail = p.get();
			tail->next = std::move(p);
			tail = new_tail;
		}
		//std::lock_guard<std::mutex> tail_lock(tail_mutex);
		data_cond.notify_one();
	}

	bool empty()
	{
		std::lock_guard<std::mutex> head_lock(head_mutex);
		return (head->next == nullptr|| head->next == NULL);
	}

};
#pragma endregion

struct MatchMark
{
	int ImageID;
	int MatchID;
	MatchMark(int a, int b) : ImageID(a), MatchID(b) {}
	MatchMark() : ImageID(-1), MatchID(-1) {}
};

void findHomography(float* inter_range,
	float* lpsm, int lpsm_size, float* lpsn, int lpsn_size, int* plMap, float* lm, float* ln,
	cv::Mat CM_Mf, cv::Mat CN_Mf, cv::Mat Ae_Mf, cv::Mat F_Mf, float* M1Trans, float* C1Trans,
	int imr, int imc, cv::Mat& match_H, cv::Mat& match_plane, cv::Mat& plane_line, float error_max, float max_ang, int index);

void findPLHomography(
	cv::Mat ppl1, cv::Mat ppl2, int* plMap, float* lm, float* ln,
	cv::Mat lmrange,
	cv::Mat CM_Mf, cv::Mat CN_Mf, cv::Mat Ae_Mf, cv::Mat F_Mf, float* M1Trans, float* C1Trans,
	int imr, int imc, cv::Mat& match_H, cv::Mat& match_plane, cv::Mat& plane_line, float error_max, float max_ang, int index);

void waitAndExecute(int thread_index);



void line2KDtree(cv::Mat lines_Mf, cv::Mat homo_Mf, cv::Mat* inter_knn_Mi, int support_H_num);
void createMap(int imr, int imc, cv::Mat inter_lines_Mf, cv::Mat lines_Mf,
	cv::Mat& inter_map_);

void createMap(int imr, int imc, cv::Mat lines_Mf,
	cv::Mat& inter_map_);

void guidedMatching(
	MatchManager* match, int mid1, int mid2, int task_id,
	float* lines1, float* lines_range, int* lines1_knn,
	float* lines2, cv::Mat line2_map,
	int lsize1, int lsize2,
	float* homos, float* planes, ushort* planes_line_id, int  hom_size,
	cv::Mat  CM, cv::Mat  CN, cv::Mat  F,
	float* M1, float* C1, float* M2, float* C2,
	int knn_num, float dist,
	int imr, int imc);
void matchPair(
	SfMManager* sfm,
	MatchManager* match,
	int match_ind,
	float error_max,
	float ang_max);


