#ifndef DL_CPU_TIMER_H
#define DL_CPU_TIMER_H

#include <sys/time.h>


namespace dl {

class DlCpuTimer
{
public:
    DlCpuTimer();
    ~DlCpuTimer();

    void start();
    void stop();
    int total_count();
    float last_elapsed();
    float total_elapsed();
    // static DlCpuTimer& getTimer();

private:
    struct timeval m_start_time;
    int m_count;
    float m_last_time;
    float m_total_time;
    bool m_is_on_timing;
};
}

#endif