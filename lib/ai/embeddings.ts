import { supabaseAdmin } from '@/lib/supabase/admin';

// Hugging Face API Configuration
const HF_API_KEY = process.env.HUGGINGFACE_API_KEY;
const HF_MODEL = 'BAAI/bge-base-en-v1.5'; // 768 dimensions (same as Xenova)

/**
 * Generate embedding using Hugging Face Inference API
 * @param text - Text to generate embedding for
 * @returns 768-dimensional embedding vector
 */
export async function generateEmbedding(text: string): Promise<number[]> {
  if (!HF_API_KEY) {
    throw new Error('HUGGINGFACE_API_KEY environment variable is not set');
  }

  try {
    console.log(`Generating embedding via Hugging Face API...`);
    
    const response = await fetch(
      `https://api-inference.huggingface.co/pipeline/feature-extraction/${HF_MODEL}`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${HF_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inputs: text,
          options: {
            wait_for_model: true, // Wait if model is loading
          },
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Hugging Face API error (${response.status}): ${errorText}`);
    }

    const result = await response.json();
    
    // API returns array of arrays for batch, get first result
    const embedding = Array.isArray(result[0]) ? result[0] : result;

    // Verify dimension
    if (embedding.length !== 768) {
      throw new Error(`Expected 768 dimensions, got ${embedding.length}`);
    }

    console.log(`✓ Embedding generated (${embedding.length} dimensions)`);
    
    return embedding;

  } catch (error) {
    console.error('Hugging Face embedding generation error:', error);
    throw error;
  }
}

// Generate embedding for user skills (cached)
let cachedUserSkillsEmbedding: number[] | null = null;

export async function generateUserSkillsEmbedding(): Promise<number[]> {
  // Return cached version if available
  if (cachedUserSkillsEmbedding) {
    console.log('Using cached user skills embedding');
    return cachedUserSkillsEmbedding;
  }

  // Query user preferences
  const { data: prefs, error } = await supabaseAdmin
    .from('user_preferences')
    .select('skills')
    .single();

  if (error || !prefs || !prefs.skills.length) {
    throw new Error('User preferences not set. Please configure your skills in the database.');
  }

  // Convert skills array to text
  const skillsText = prefs.skills.join(', ');
  console.log('Generating embedding for user skills:', skillsText);

  // Generate embedding
  cachedUserSkillsEmbedding = await generateEmbedding(skillsText);
  
  return cachedUserSkillsEmbedding;
}

// Clear cache (useful for testing)
export function clearUserSkillsCache() {
  cachedUserSkillsEmbedding = null;
}
