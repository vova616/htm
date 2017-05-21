use std::collections::VecDeque;
use std;
use numext::ClipExt;

pub struct SDRClassifier<T> where T: std::clone::Clone {
    
    /**
     * The alpha used to adapt the weight matrix during
     * learning. A larger alpha results in faster adaptation to the data.
     */
    alpha :f32, //  : 0.001,
    /**
	 * Used to track the actual value within each
     * bucket. A lower actValueAlpha results in longer term memory
	 */
    act_value_alpha: f32, // : 0.3;
    /** 
     * The bit's learning iteration. This is updated each time store() gets
     * called on this bit.
     */
    learn_iteration: u32,
    /**
     * This contains the offset between the recordNum (provided by caller) and
     * learnIteration (internal only, always starts at 0).
     */
    record_num_minus_learn_iteration: i32,  //= -1;
	/**
	 * This contains the highest value we've ever seen from the list of active cell indexes
	 * from the TM (patternNZ). It is used to pre-allocate fixed size arrays that holds the weights.
	 */
	max_input_idx: usize,  // = 0;

    /**
     * This contains the value of the highest bucket index we've ever seen
     * It is used to pre-allocate fixed size arrays that hold the weights of
     * each bucket index during inference 
     */
    max_bucket_idx: usize,

	/**
	 * The connection weight matrix
	 */
    weight_matrix: Vec<Vec<Vec<f32>>>,
    /** The sequence different steps of multi-step predictions */
    steps: Vec<u8>,
    /**
     * History of the last _maxSteps activation patterns. We need to keep
     * these so that we can associate the current iteration's classification
     * with the activationPattern from N steps ago
     */
    pattern_history: VecDeque<(u32, Vec<usize>)>,
    /**
     * This keeps track of the actual value to use for each bucket index. We
     * start with 1 bucket, no actual value so that the first infer has something
     * to return
     */
    actual_values: Vec<Option<T>>,
}

impl<T> SDRClassifier<T> where T: std::clone::Clone {

    /**
     * Constructor for the SDRClassifier
     * 
     * @param steps Sequence of the different steps of multi-step predictions to learn.
     * @param alpha The alpha used to adapt the weight matrix during learning. A larger alpha
     * 		  results in faster adaptation to the data.
     * @param actValueAlpha Used to track the actual value withing each bucket. A lower 
     * 		  actValueAlpha results in longer term memory.
     * @param verbosity Verbosity level, can be 0, 1, or 2.
     */
	pub fn new(steps: Vec<u8>, alpha: f32, act_value_alpha: f32, column_size: usize) -> SDRClassifier<T> {
        let max = *steps.iter().max().unwrap() as usize + 1;
        let len = steps.len();
        SDRClassifier {
            steps: steps,
            alpha: alpha,
            act_value_alpha: act_value_alpha,
            actual_values: vec![None],
            pattern_history: VecDeque::with_capacity(max),
            record_num_minus_learn_iteration: -1,
            max_bucket_idx: 0,
            max_input_idx: column_size - 1,
            learn_iteration: 0,
            weight_matrix: vec![vec![vec![0f32; column_size]; 1]; len]
        }
	}

    /**
	 * Process one input sample.
	 * This method is called by outer loop code outside the nupic-engine. We 
	 * use this instead of the nupic engine compute() because our inputs and 
	 * outputs aren't fixed size vectors of reals.
	 * <p>
	 * @param recordNum <p>
	 * Record number of this input pattern. Record numbers normally increase
	 * sequentially by 1 each time unless there are missing records in the
	 * dataset. Knowing this information ensures that we don't get confused by
	 * missing records.
	 * @param classification <p>
	 * {@link Map} of the classification information:
	 * <p>&emsp;"bucketIdx" - index of the encoder bucket
	 * <p>&emsp;"actValue" -  actual value doing into the encoder
	 * @param patternNZ <p>
	 * List of the active indices from the output below. When the output is from
	 * the TemporalMemory, this array should be the indices of the active cells.
	 * @param learn <p>
	 * If true, learn this sample.
	 * @param infer <p>
	 * If true, perform inference. If false, null will be returned.
	 * 
	 * @return
	 * {@link Classification} containing inference results if {@code learn} param is true,
	 * otherwise, will return {@code null}. The Classification
	 * contains the computed probability distribution (relative likelihood for each
	 * bucketIdx starting from bucketIdx 0) for each step in {@code steps}. Each bucket's
	 * likelihood can be accessed individually, or all the buckets' likelihoods can
	 * be obtained in the form of a double array.
	 *
 	 * <pre>{@code
 	 * //Get likelihood val for bucket 0, 5 steps in future
	 * classification.getStat(5, 0);
	 *
 	 * //Get all buckets' likelihoods as double[] where each
	 * //index is the likelihood for that bucket
	 * //(e.g. [0] contains likelihood for bucketIdx 0)
	 * classification.getStats(5);
	 * }</pre>
	 *
	 * The Classification also contains the average actual value for each bucket.
	 * The average values for the buckets can be accessed individually, or altogether
	 * as a double[].
	 *
	 * <pre>{@code
	 * //Get average actual val for bucket 0
	 * classification.getActualValue(0);
	 *
	 * //Get average vals for all buckets as double[], where
	 * //each index is the average val for that bucket
	 * //(e.g. [0] contains average val for bucketIdx 0)
	 * classification.getActualValues();
	 * }</pre>
	 *
	 * The Classification can also be queried for the most probable bucket (the bucket
	 * with the highest associated likelihood value), as well as the average input value
	 * that corresponds to that bucket.
	 *
	 * <pre>{@code
	 * //Get index of most probable bucket
	 * classification.getMostProbableBucketIndex();
	 *
	 * //Get the average actual val for that bucket
	 * classification.getMostProbableValue();
	 * }</pre>
	 *
	 */
    pub fn compute(&mut self, record_num: u32, bucket_idx: usize, act_value: T, pattern: &[usize], learn: bool, infer: bool) -> Vec<(u8, Vec<f32>)> {
       // Classification<T> retVal = null;
        //List<T> actualValues = (List<T>)this.actualValues;

		//Save the offset between recordNum and learnIteration if this is the first compute
		if self.record_num_minus_learn_iteration == -1 {
			self.record_num_minus_learn_iteration = record_num as i32 - self.learn_iteration as i32;
        }

		//Update the learn iteration
		self.learn_iteration = (record_num as i32 - self.record_num_minus_learn_iteration) as u32;

		//Store pattern in our history
        if self.pattern_history.len() == self.pattern_history.capacity() {
            self.pattern_history.pop_back();
        }
		self.pattern_history.push_front((self.learn_iteration, pattern.to_vec()));

		
		//------------------------------------------------------------------------
		//Inference:
		//For each active bit in the activationPattern, get the classification votes
        let ret = if infer  {
			self.infer(pattern)
		} else {
            vec![(0, vec![0.0])]
        };

		//------------------------------------------------------------------------
		//Learning:
		if learn {
			// Update maxBucketIndex and augment weight matrix with zero padding
			if bucket_idx > self.max_bucket_idx  {
				for &steps in &self.steps {
					for i in self.max_bucket_idx..bucket_idx {
                        self.weight_matrix[steps as usize].push(vec![0.0; self.max_input_idx + 1]);
					}
				}
				self.max_bucket_idx = bucket_idx;
			}


			// Update rolling average of actual values if it's a scalar. If it's not, it
			// must be a category, in which case each bucket only ever sees on category so
			// we don't need a running average.
			while self.max_bucket_idx > self.actual_values.len() - 1 {
				self.actual_values.push(None);
			}

        
            if self.actual_values[bucket_idx].is_none() {
                self.actual_values[bucket_idx] = Some(act_value);
            } else {
                // let value = self.actual_values[bucket_idx].unwrap();
                //TODO: find a generic way to deal with numbers
                /*
                Double val = ((1.0 - actValueAlpha) * ((Number)actualValues.get(bucketIdx)).doubleValue() +
							actValueAlpha * ((Number)actValue).doubleValue());
					actualValues.set(bucketIdx, (T)val);
                */
                self.actual_values[bucket_idx] = Some(act_value);
            }
            
            let mut error = vec![0f32; self.max_bucket_idx + 1];

            for &(ref iter,ref pattern) in &self.pattern_history {
                let nSteps = (self.learn_iteration - iter) as usize;
                let nstps = nSteps as u8;
                if self.steps.contains(&nstps) {
                    self.infer_single_step(&pattern, nSteps, &mut error);
                    for (index, val) in error.iter_mut().enumerate() {
                        *val = ((index == bucket_idx) as usize) as f32 - *val;
                    }
                    for (index, matrix) in self.weight_matrix[nSteps].iter_mut().enumerate() {
                        for &bit in pattern {
                            matrix[bit] += self.alpha * error[index];
                            //matrix[bit].clip(-1.0, 1.0); not sure if needed
                        }
                    }
                }
			}
		}   
        

		ret
    }

    pub fn infer(&self, pattern: &[usize]) -> Vec<(u8, Vec<f32>)> {
        self.steps.iter().map(|&step| {
            let mut error = vec![0f32; self.max_bucket_idx + 1];
            self.infer_single_step(pattern, step as usize, &mut error[..]);
			(step, error)
        }).collect()
	}

    pub fn infer_single_step(&self, pattern: &[usize], step: usize, into: &mut [f32]) {
		// Compute the output activation "level" for each bucket (matrix row)
		// we've seen so far and store in double[]

        let matrix = &self.weight_matrix[step];

        for (index,val) in into.iter_mut().enumerate() {
            *val = 0.0;
            for &pattern_value in pattern {
                *val += matrix[index][pattern_value];
            }
        }

        let mut sum = 0.0;
        for val in into.iter_mut() {
            if *val < 0.001 {
                *val = 0.0;
            } else {
                *val *= *val;
            }
            sum += *val;
        }

        if sum > 0.001 {
            for val in into.iter_mut() { 
                *val /= sum;
            }
        }
	}
    
}