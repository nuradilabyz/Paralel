import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
public class Task4 {
    static void item(int i) {
        long nanos=100_000L*(i%20);
        if(nanos==0)return;
        try {Thread.sleep(nanos/1_000_000,(int)(nanos%1_000_000));}
        catch(InterruptedException e) {Thread.currentThread().interrupt();throw new RuntimeException(e);}
    }
    static void run(boolean dynamic) throws Exception {
        final int workers=4,n=1000;
        ExecutorService pool=Bench.pool(workers);
        CountDownLatch ready=new CountDownLatch(workers),go=new CountDownLatch(1);
        AtomicInteger next=new AtomicInteger(),done=new AtomicInteger(); long[] finish=new long[workers];
        java.util.ArrayList<Future<?>> jobs=new java.util.ArrayList<>();
        try {
            for(int w=0;w<workers;w++) { final int k=w;jobs.add(pool.submit(()->{
                ready.countDown();try {go.await();}catch(InterruptedException e){throw new RuntimeException(e);}
                if(dynamic) {int i;while((i=next.getAndIncrement())<n){item(i);done.incrementAndGet();}}
                else for(int i=k*250;i<(k+1)*250;i++){item(i);done.incrementAndGet();}
                finish[k]=System.nanoTime();
            })); }
            ready.await();long s=System.nanoTime();go.countDown();for(Future<?> j:jobs)j.get();
            double elapsed=Bench.ms(s);Bench.check(done.get()==1000,"Missing work");
            long min=java.util.Arrays.stream(finish).min().orElseThrow(),max=java.util.Arrays.stream(finish).max().orElseThrow();
            System.out.printf("%s | Total %.6f ms | First finish %.6f ms | Last finish %.6f ms | Delta %.6f ms%n",dynamic?"Dynamic queue (chunk=1)":"Static equal blocks",elapsed,(min-s)/1e6,(max-s)/1e6,(max-min)/1e6);
        } finally {Bench.close(pool);}
    }
    public static void main(String[] args) throws Exception {
        Bench.header("TASK 4 - Scheduling under skew");
        System.out.println("1000 items, 4 workers, sleep(0.0001 * (i % 20)) seconds. Three paired runs:");
        for(int r=1;r<=3;r++){System.out.println("Run "+r);run(false);run(true);}
        System.out.println("Periodic costs are nearly balanced in contiguous blocks; dynamic scheduling is not guaranteed to win. Sleep duration depends on OS scheduling.");
    }
}
