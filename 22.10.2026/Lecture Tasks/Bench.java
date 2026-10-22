import java.util.*;
import java.util.concurrent.*;
import java.util.function.IntConsumer;
import java.io.*;

/** Shared timing, hardware and deterministic input helpers. No external dependencies. */
public final class Bench {
    static final long STUDENT_ID = Long.getLong("student.id", 230103188L);
    static final int CORES = Runtime.getRuntime().availableProcessors();
    static volatile double sink;
    static void header(String title) throws Exception {
        Locale.setDefault(Locale.US);
        System.out.println(title);
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.version"));
        System.out.println("Architecture: " + System.getProperty("os.arch"));
        if (System.getProperty("os.name").contains("Mac")) {
            Process proc = new ProcessBuilder("sysctl", "-n", "machdep.cpu.brand_string").start();
            System.out.println("CPU: " + new String(proc.getInputStream().readAllBytes()).trim());
            proc.waitFor();
        }
        System.out.println("Logical cores: " + CORES);
        System.out.println("Java: " + System.getProperty("java.version"));
    }
    static double[] input(int n) {
        double[] a = new double[n];
        SplittableRandom r = new SplittableRandom(STUDENT_ID);
        for (int i=0; i<n; i++) a[i]=r.nextDouble();
        return a;
    }
    static double ms(long start) { return (System.nanoTime()-start)/1e6; }
    interface Action { void run() throws Exception; }
    static double median(Action action, int warmups, int repeats) throws Exception {
        for(int i=0;i<warmups;i++) action.run();
        double[] t=new double[repeats];
        for(int i=0;i<repeats;i++) { long s=System.nanoTime(); action.run(); t[i]=ms(s); }
        Arrays.sort(t); return t[repeats/2];
    }
    static void parallel(ExecutorService pool,int workers,int n,IntConsumer range) throws Exception {
        ArrayList<Future<?>> jobs=new ArrayList<>();
        for(int w=0;w<workers;w++) { final int k=w; jobs.add(pool.submit(()->range.accept(k))); }
        for(Future<?> job:jobs) job.get();
    }
    static ExecutorService pool(int n) throws Exception {
        ThreadPoolExecutor pool=(ThreadPoolExecutor)Executors.newFixedThreadPool(n);
        pool.prestartAllCoreThreads(); return pool;
    }
    static void close(ExecutorService pool) throws Exception {
        pool.shutdown(); if(!pool.awaitTermination(1,TimeUnit.MINUTES)) throw new IllegalStateException("Pool timeout");
    }
    static void check(boolean ok,String message) { if(!ok) throw new AssertionError(message); }
}
