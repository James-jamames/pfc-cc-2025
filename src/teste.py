import multiprocessing as mp
import math

# Função de processamento (cada worker roda essa função)
def process_batch(batch):
    resultado = [x for x in batch]  # exemplo: elevar cada valor ao quadrado
    return resultado

if __name__ == "__main__":
    # Conjunto de dados
    data = list(range(1, 101))  # de 1 a 100
    
    # Número de núcleos disponíveis
    n_cores = mp.cpu_count()
    print(f"Usando {n_cores} núcleos")

    # Quebrar a lista em batches
    batch_size = math.ceil(len(data) / n_cores)
    batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    # Criar pool de processos
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(process_batch, batches)

    # Juntar resultados
    final_result = [item for sublist in results for item in sublist]
    print(final_result)