import os
import psutil
import time
from snp_metrics_old.snp_metrics import process_idat_files
import logging
import threading
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_system_metrics():
    """Collect system-wide metrics."""
    return {
        'timestamp': time.time(),
        'system_cpu': psutil.cpu_percent(interval=None),
        'system_memory': psutil.virtual_memory().percent,
        'swap_memory': psutil.swap_memory().percent,
        'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None,
        'net_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None
    }

def monitor_with_callback(process_func, *args, **kwargs):
    """Wrapper to monitor CPU and memory usage across all related processes."""
    main_process = psutil.Process()
    process_measurements = []
    system_measurements = []
    
    def collect_metrics():
    
        all_processes = [main_process] + main_process.children(recursive=True)
        
        total_cpu = 0
        total_memory = 0
        total_threads = 0
        
        for proc in all_processes:
            try:
                total_cpu += proc.cpu_percent()
                total_memory += proc.memory_percent()
                total_threads += proc.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process might have terminated
                continue
        
        process_measurements.append({
            'timestamp': time.time(),
            'cpu_percent': total_cpu,
            'memory_percent': total_memory,
            'num_threads': total_threads,
            'num_processes': len(all_processes)
        })
        
        # Collect system-wide metrics
        system_measurements.append(collect_system_metrics())
    
    # Start monitoring thread
    stop_monitoring = threading.Event()
    
    def log_process_tree():
        children = main_process.children(recursive=True)
        logger.info(f"Current process tree: Main PID={main_process.pid}, "
                   f"Children PIDs={[p.pid for p in children]}")
    
    def monitoring_loop():
        while not stop_monitoring.is_set():
            collect_metrics()
            log_process_tree()  # Log process tree every iteration
            time.sleep(1)  # Collect metrics every second
    
    # Initialize CPU monitoring
    for proc in [main_process] + main_process.children(recursive=True):
        try:
            proc.cpu_percent()  # First call to initialize CPU monitoring
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    monitor_thread = threading.Thread(target=monitoring_loop)
    monitor_thread.start()
    
    try:
        result = process_func(*args, **kwargs)
        return result, process_measurements, system_measurements
    finally:
        stop_monitoring.set()
        monitor_thread.join()

def main():
    # Hardcoded paths
    idat_path = "/home/vitaled2/gp2-microservices/services/idat_utils/data/207847320055"
    bpm = "/home/vitaled2/gp2-microservices/services/idat_utils/data/ilmn_utils/NeuroBooster_20042459_A2.bpm"
    bpm_csv = "/home/vitaled2/gp2-microservices/services/idat_utils/data/ilmn_utils/NeuroBooster_20042459_A2.csv"
    egt = "/home/vitaled2/gp2-microservices/services/idat_utils/data/ilmn_utils/recluster_09092022.egt"
    ref_fasta = "/home/vitaled2/gp2-microservices/services/idat_utils/data/ref/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
    output_directory = "/home/vitaled2/gp2-microservices/services/idat_utils/data/output/snp_metrics"
    bcftools_plugins_path = "/home/vitaled2/gp2-microservices/services/idat_utils/bin"
    iaap = "/home/vitaled2/gp2-microservices/services/idat_utils/exec/iaap-cli-linux-x64-1.1.0-sha.80d7e5b3d9c1fdfc2e99b472a90652fd3848bbc7/iaap-cli/iaap-cli"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Processing IDAT directory: {idat_path}")
    
    # Wrap process_idat_files with our monitoring function
    success, process_measurements, system_measurements = monitor_with_callback(
        process_idat_files,
        idat_path=idat_path,
        output_directory=output_directory,
        bpm=bpm,
        bpm_csv=bpm_csv,
        egt=egt,
        ref_fasta=ref_fasta,
        iaap=iaap,
        bcftools_plugins_path=bcftools_plugins_path,
        cleanup_intermediate_files=True
    )
    
    # Enhanced performance metrics output
    if process_measurements:
        cpu_usage = [m['cpu_percent'] for m in process_measurements]
        memory_usage = [m['memory_percent'] for m in process_measurements]
        process_counts = [m['num_processes'] for m in process_measurements]
        
        print(f"\nPerformance Metrics:")
        print(f"Average CPU: {sum(cpu_usage)/len(cpu_usage):.2f}%")
        print(f"Peak CPU: {max(cpu_usage):.2f}%")
        print(f"Min CPU: {min(cpu_usage):.2f}%")
        print(f"Average Memory: {sum(memory_usage)/len(memory_usage):.2f}%")
        print(f"Peak Memory: {max(memory_usage):.2f}%")
        print(f"Average Process Count: {sum(process_counts)/len(process_counts):.1f}")
        print(f"Max Process Count: {max(process_counts)}")
        
        import json
        with open('performance_metrics.json', 'w') as f:
            json.dump(process_measurements, f)
    
    if system_measurements:
        system_cpu_usage = [m['system_cpu'] for m in system_measurements]
        system_memory_usage = [m['system_memory'] for m in system_measurements]
        system_swap_usage = [m['swap_memory'] for m in system_measurements]
        system_disk_io = [m['disk_io'] for m in system_measurements if m['disk_io']]
        system_net_io = [m['net_io'] for m in system_measurements if m['net_io']]
        
        print(f"\nSystem-Wide Metrics:")
        print(f"Average System CPU: {sum(system_cpu_usage)/len(system_cpu_usage):.2f}%")
        print(f"Peak System CPU: {max(system_cpu_usage):.2f}%")
        print(f"Average System Memory: {sum(system_memory_usage)/len(system_memory_usage):.2f}%")
        print(f"Peak System Memory: {max(system_memory_usage):.2f}%")
        print(f"Average System Swap: {sum(system_swap_usage)/len(system_swap_usage):.2f}%")
        print(f"Peak System Swap: {max(system_swap_usage):.2f}%")
        
        if system_disk_io:
            print(f"\nDisk I/O Metrics:")
            for metric, value in system_disk_io[0].items():
                print(f"{metric}: {value}")
        
        if system_net_io:
            print(f"\nNetwork I/O Metrics:")
            for metric, value in system_net_io[0].items():
                print(f"{metric}: {value}")
        
        with open('system_metrics.json', 'w') as f:
            json.dump(system_measurements, f)
    
    if success:
        print("IDAT processing completed successfully")
        barcode = os.path.basename(idat_path)
        tmp_dir = os.path.join(output_directory, f"tmp_{barcode}")
        print(f"Intermediate GTC files are stored in: {tmp_dir}")
        print(f"Intermediate VCF files are stored in: {tmp_dir}")
    else:
        print("IDAT processing failed")

if __name__ == "__main__":
    main() 