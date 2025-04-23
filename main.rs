use std::error::Error;
use std::fs::File;
use std::io::Read;
use ndarray::{Array1, Array2, Axis, s};
use csv::ReaderBuilder;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Program Prediksi Harga Cabai dengan SVM dan K-NN");
    
    // Load data
    let data = load_data("harga_cabai_johar_2016-2023.csv")?;
    
    // Ekstrak fitur waktu dan target
    let (features, targets) = prepare_data(&data);
    
    // Split data menjadi training dan testing set (80% training, 20% testing)
    let test_size = features.nrows() / 5;
    let train_size = features.nrows() - test_size;
    
    let train_features = features.slice(s![0..train_size, ..]).to_owned();
    let train_targets = targets.slice(s![0..train_size]).to_owned();
    let test_features = features.slice(s![train_size.., ..]).to_owned();
    let test_targets = targets.slice(s![train_size..]).to_owned();
    
    // Implementasi manual SVM (pendekatan sederhana berbasis regresi)
    println!("\n=== Metode Support Vector Machine (SVM) ===");
    let svm_predictions = simple_svm_regression(&train_features, &train_targets, &test_features);
    
    // Implementasi manual K-NN
    println!("\n=== Metode K-Nearest Neighbor (K-NN) ===");
    let knn_predictions = knn_regression(&train_features, &train_targets, &test_features, 5);
    
    // Hitung dan tampilkan RMSE
    let test_targets_vec = test_targets.to_vec();
    let svm_rmse = calculate_rmse(&test_targets_vec, &svm_predictions);
    let knn_rmse = calculate_rmse(&test_targets_vec, &knn_predictions);
    
    println!("RMSE SVM: {:.2}", svm_rmse);
    println!("RMSE K-NN: {:.2}", knn_rmse);
    
    // Tampilkan perbandingan hasil prediksi
    println!("\nPerbandingan Data Aktual vs Prediksi:");
    println!("{:<5} {:<12} {:<12} {:<12}", "No", "Aktual", "SVM", "K-NN");
    println!("{:-<45}", "");
    for i in 0..test_targets_vec.len() {
        println!("{:<5} {:<12.2} {:<12.2} {:<12.2}", 
                i+1, test_targets_vec[i], svm_predictions[i], knn_predictions[i]);
    }
    
    // Plot hasil perbandingan
    plot_comparisons("svm_comparison.png", &test_targets_vec, &svm_predictions, "SVM")?;
    plot_comparisons("knn_comparison.png", &test_targets_vec, &knn_predictions, "K-NN")?;
    
    Ok(())
}

/// Fungsi untuk memuat data dari file CSV
fn load_data(filename: &str) -> Result<Vec<(String, f64)>, Box<dyn Error>> {
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(contents.as_bytes());
    
    let mut data = Vec::new();
    
    for result in reader.records() {
        let record = result?;
        let periode = record.get(1).unwrap_or("").to_string();
        let harga: f64 = record.get(2).unwrap_or("0").parse()?;
        
        data.push((periode, harga));
    }
    
    println!("Data berhasil dimuat: {} entri", data.len());
    Ok(data)
}

/// Fungsi untuk menyiapkan data menjadi fitur dan target
fn prepare_data(data: &[(String, f64)]) -> (Array2<f64>, Array1<f64>) {
    // Ekstrak fitur waktu sebagai dua komponen: tahun dan bulan
    let mut features = Array2::zeros((data.len(), 3));
    let mut targets = Array1::zeros(data.len());
    
    // Fitur tambahan: harga bulan sebelumnya
    let mut prev_price = data[0].1; // Inisialisasi dengan harga awal
    
    for (i, (periode, harga)) in data.iter().enumerate() {
        // Parse periode (format: MMM-YYYY)
        let parts: Vec<&str> = periode.split('-').collect();
        
        // Ekstrak bulan
        let month = match parts[0] {
            "Jan" => 1.0, "Feb" => 2.0, "Mar" => 3.0, "Apr" => 4.0,
            "May" => 5.0, "Jun" => 6.0, "Jul" => 7.0, "Aug" => 8.0,
            "Sep" => 9.0, "Oct" => 10.0, "Nov" => 11.0, "Dec" => 12.0,
            _ => 0.0,
        };
        
        // Ekstrak tahun dan konversi ke angka (2016 -> 0, 2017 -> 1, dst)
        let year: f64 = parts[1].parse::<f64>().unwrap_or(0.0) - 2016.0;
        
        // Simpan fitur
        features[[i, 0]] = year;
        features[[i, 1]] = month;
        features[[i, 2]] = prev_price;
        
        // Simpan target (harga)
        targets[i] = *harga;
        
        // Update harga sebelumnya untuk entri berikutnya
        prev_price = *harga;
    }
    
    // Normalisasi fitur
    for col in 0..features.ncols() {
        let col_data = features.column(col);
        let min_val = col_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = col_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if max_val > min_val {
            for i in 0..features.nrows() {
                features[[i, col]] = (features[[i, col]] - min_val) / (max_val - min_val);
            }
        }
    }
    
    (features, targets)
}

/// Implementasi sederhana regresi linier untuk menyimulasikan SVM
fn simple_svm_regression(
    train_features: &Array2<f64>, 
    train_targets: &Array1<f64>, 
    test_features: &Array2<f64>,
) -> Vec<f64> {
    println!("Melatih model SVM (implementasi regresi linier sederhana)...");
    
    // Vektor untuk menyimpan hasil prediksi
    let mut predictions = Vec::with_capacity(test_features.nrows());
    
    // Implementasi sederhana regresi linier dengan optimasi gradien
    let mut weights = vec![0.0; train_features.ncols()];
    let mut bias = 0.0;
    
    // Parameter pembelajaran
    let learning_rate = 0.01;
    let iterations = 1000;
    
    // Pelatihan model
    for _ in 0..iterations {
        // Inisialisasi gradien
        let mut weight_gradients = vec![0.0; weights.len()];
        let mut bias_gradient = 0.0;
        
        // Untuk setiap data latih
        for i in 0..train_features.nrows() {
            // Hitung prediksi
            let mut prediction = bias;
            for j in 0..weights.len() {
                prediction += weights[j] * train_features[[i, j]];
            }
            
            // Hitung error
            let error = prediction - train_targets[i];
            
            // Akumulasi gradien
            for j in 0..weights.len() {
                weight_gradients[j] += error * train_features[[i, j]];
            }
            bias_gradient += error;
        }
        
        // Update parameter dengan nilai gradien rata-rata
        let n = train_features.nrows() as f64;
        for j in 0..weights.len() {
            weights[j] -= learning_rate * weight_gradients[j] / n;
        }
        bias -= learning_rate * bias_gradient / n;
    }
    
    // Prediksi pada data uji
    for i in 0..test_features.nrows() {
        let mut prediction = bias;
        for j in 0..weights.len() {
            prediction += weights[j] * test_features[[i, j]];
        }
        predictions.push(prediction);
    }
    
    println!("Hasil Prediksi SVM:");
    for (i, &pred) in predictions.iter().enumerate() {
        println!("Data ke-{}: Prediksi = {:.2}", i+1, pred);
    }
    
    predictions
}

/// Implementasi K-Nearest Neighbor untuk regresi
fn knn_regression(
    train_features: &Array2<f64>, 
    train_targets: &Array1<f64>, 
    test_features: &Array2<f64>,
    k: usize,
) -> Vec<f64> {
    println!("Melatih model K-Nearest Neighbor dengan k = {}...", k);
    
    // Vektor untuk menyimpan hasil prediksi
    let mut predictions = Vec::with_capacity(test_features.nrows());
    
    // Prediksi untuk setiap data uji
    for test_idx in 0..test_features.nrows() {
        let test_point = test_features.row(test_idx);
        
        // Hitung jarak Euclidean dengan setiap data latih
        let mut distances = Vec::with_capacity(train_features.nrows());
        
        for train_idx in 0..train_features.nrows() {
            let train_point = train_features.row(train_idx);
            
            // Hitung jarak Euclidean
            let mut squared_dist = 0.0;
            for j in 0..train_features.ncols() {
                let diff = test_point[j] - train_point[j];
                squared_dist += diff * diff;
            }
            let distance = squared_dist.sqrt();
            
            distances.push((distance, train_targets[train_idx]));
        }
        
        // Urutkan jarak dari terdekat ke terjauh
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Pilih k tetangga terdekat
        let mut prediction_sum = 0.0;
        for i in 0..k {
            prediction_sum += distances[i].1;
        }
        
        // Prediksi adalah rata-rata nilai target dari k tetangga terdekat
        let prediction = prediction_sum / (k as f64);
        predictions.push(prediction);
    }
    
    println!("Hasil Prediksi K-NN:");
    for (i, &pred) in predictions.iter().enumerate() {
        println!("Data ke-{}: Prediksi = {:.2}", i+1, pred);
    }
    
    predictions
}

/// Fungsi untuk menghitung Root Mean Square Error (RMSE)
fn calculate_rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    let mut sum_squared_error = 0.0;
    let n = actual.len();
    
    for i in 0..n {
        let error = actual[i] - predicted[i];
        sum_squared_error += error * error;
    }
    
    (sum_squared_error / n as f64).sqrt()
}

/// Fungsi untuk membuat plot perbandingan data aktual vs prediksi
fn plot_comparisons(
    filename: &str, 
    actual: &[f64], 
    predicted: &[f64], 
    method: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Tentukan rentang nilai untuk plot
    let min_value = actual.iter()
        .chain(predicted.iter())
        .fold(f64::INFINITY, |a, &b| a.min(b))
        .min(0.0); // Pastikan dimulai dari 0 atau lebih rendah
    
    let max_value = actual.iter()
        .chain(predicted.iter())
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        .max(100000.0) * 1.1; // Berikan sedikit ruang di atas
    
    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Perbandingan Harga Aktual vs Prediksi ({})", method), ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(
            0..actual.len(),
            min_value..max_value,
        )?;
    
    chart.configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| format!("{}", x))
        .y_label_formatter(&|y| format!("{:.0}", y))
        .x_desc("Data ke-")
        .y_desc("Harga (Rp)")
        .draw()?;
    
    // Plot data aktual
    chart.draw_series(LineSeries::new(
        (0..actual.len()).map(|i| (i, actual[i])),
        &BLUE,
    ))?
    .label("Harga Aktual")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    
    // Plot data prediksi
    chart.draw_series(LineSeries::new(
        (0..predicted.len()).map(|i| (i, predicted[i])),
        &RED,
    ))?
    .label("Harga Prediksi")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    println!("Plot perbandingan {} disimpan ke: {}", method, filename);
    
    Ok(())
}