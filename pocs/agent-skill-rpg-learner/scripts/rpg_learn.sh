#!/usr/bin/env bash
set -u

skill_name="terminal-rpg-agent"
state_dir="${RPG_STATE_DIR:-$HOME/.terminal-rpg-agent/state}"
theme_input="${1:-}"
score=0
xp=0
level=1
health=100
max_health=100
streak=0
hint_tokens=1
shields=1
double_xp=0
retry_charm=1
boss_key=0
boss_health=100
boss_defeated=false
quest_index=0
auto_index=0
awarded_item=""
input_value=""
auto_answers=()

if [ -n "${RPG_AUTO_ANSWERS:-}" ]; then
  IFS=',' read -r -a auto_answers <<< "$RPG_AUTO_ANSWERS"
fi

supports_color=false
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  supports_color=true
fi

paint() {
  code="$1"
  text="$2"
  if [ "$supports_color" = true ]; then
    printf '\033[%sm%s\033[0m' "$code" "$text"
  else
    printf '%s' "$text"
  fi
}

sleep_step() {
  if [ "${RPG_FAST:-0}" != "1" ] && [ -t 1 ]; then
    sleep 0.15
  fi
}

line() {
  printf '+------------------------------------------------------------------------------+\n'
}

card_line() {
  text="$1"
  while [ "${#text}" -gt 74 ]; do
    printf '| %-74s |\n' "${text:0:74}"
    text="${text:74}"
  done
  printf '| %-74s |\n' "$text"
}

card() {
  title="$1"
  shift
  line
  card_line "$title"
  line
  for text in "$@"; do
    card_line "$text"
  done
  line
}

bar() {
  current="$1"
  total="$2"
  width="$3"
  if [ "$current" -lt 0 ]; then
    current=0
  fi
  if [ "$current" -gt "$total" ]; then
    current="$total"
  fi
  filled=$((current * width / total))
  empty=$((width - filled))
  out=""
  i=0
  while [ "$i" -lt "$filled" ]; do
    out="${out}#"
    i=$((i + 1))
  done
  i=0
  while [ "$i" -lt "$empty" ]; do
    out="${out}-"
    i=$((i + 1))
  done
  printf '[%s] %s/%s' "$out" "$current" "$total"
}

normalize_theme() {
  raw="$1"
  lower=$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')
  case "$lower" in
    1|algorithm|algorithms|algo) printf 'algorithms' ;;
    2|datastructure|datastructures|data-structure|data-structures|"data structure"|"data structures"|ds) printf 'datastructures' ;;
    3|generative-ai|"generative ai"|genai|ai) printf 'generative-ai' ;;
    4|machine-learning|"machine learning"|ml) printf 'machine-learning' ;;
    5|sre-devops|sre|devops|"sre devops"|"sre/devops") printf 'sre-devops' ;;
    6|management|manager|leadership) printf 'management' ;;
    *) printf '' ;;
  esac
}

theme_title() {
  case "$1" in
    algorithms) printf 'Algorithms' ;;
    datastructures) printf 'Data Structures' ;;
    generative-ai) printf 'Generative AI' ;;
    machine-learning) printf 'Machine Learning' ;;
    sre-devops) printf 'SRE/DevOps' ;;
    management) printf 'Management' ;;
  esac
}

choose_theme() {
  theme=$(normalize_theme "$theme_input")
  while [ -z "$theme" ]; do
    card "Choose Your Realm" "1. Algorithms" "2. Data Structures" "3. Generative AI" "4. Machine Learning" "5. SRE/DevOps" "6. Management"
    printf 'Pick 1-6: '
    read -r theme_input
    theme=$(normalize_theme "$theme_input")
  done
  printf '%s' "$theme"
}

next_input() {
  prompt="$1"
  if [ "${#auto_answers[@]}" -gt "$auto_index" ]; then
    input_value="${auto_answers[$auto_index]}"
    auto_index=$((auto_index + 1))
    printf '%s%s\n' "$prompt" "$input_value" >&2
    return
  fi
  printf '%s' "$prompt" >&2
  read -r input_value
}

choice_number() {
  raw=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  case "$raw" in
    1|a) printf '1' ;;
    2|b) printf '2' ;;
    3|c) printf '3' ;;
    4|d) printf '4' ;;
    h|hint) printf 'h' ;;
    *) printf '' ;;
  esac
}

gain_xp() {
  amount="$1"
  if [ "$double_xp" -gt 0 ]; then
    amount=$((amount * 2))
    double_xp=$((double_xp - 1))
  fi
  xp=$((xp + amount))
  level=$((xp / 100 + 1))
}

award_item() {
  item="$1"
  case "$item" in
    hint) hint_tokens=$((hint_tokens + 1)); awarded_item="Hint Token" ;;
    shield) shields=$((shields + 1)); awarded_item="Shield" ;;
    double) double_xp=$((double_xp + 1)); awarded_item="Double XP" ;;
    retry) retry_charm=$((retry_charm + 1)); awarded_item="Retry Charm" ;;
    key) boss_key=$((boss_key + 1)); awarded_item="Boss Key" ;;
    *) awarded_item="Gold" ;;
  esac
}

stats_line() {
  printf 'HP %s/%s | XP %s | LVL %s | SCORE %s | STREAK %s' "$health" "$max_health" "$xp" "$level" "$score" "$streak"
}

inventory_line() {
  printf 'Hints %s | Shields %s | Double XP %s | Retries %s | Boss Keys %s' "$hint_tokens" "$shields" "$double_xp" "$retry_charm" "$boss_key"
}

render_status() {
  card "Status" "$(stats_line)" "$(inventory_line)" "Progress $(bar "$quest_index" 5 24)"
}

algorithms_q=(
"Which complexity usually describes binary search on a sorted array?|O(n)|O(log n)|O(n log n)|O(1)|2|hint|Binary search cuts the remaining search space each step."
"A recursive function needs what to stop safely?|A base case|A larger input|A global variable|A sorted array|1|shield|Without a stopping condition, recursion keeps calling itself."
"Which sorting algorithm has average O(n log n) behavior?|Bubble sort|Insertion sort|Merge sort|Linear scan|3|double|Divide and merge is the core pattern."
"What does a greedy algorithm do at each step?|Tries every subset|Chooses a local best move|Stores all previous answers|Ignores constraints|2|retry|Greedy methods commit to a strong local choice."
"Dynamic programming is strongest when a problem has what?|Random inputs only|No repeated work|Overlapping subproblems|Only one valid path|3|key|DP reuses answers for repeated subproblems."
)

algorithms_b=(
"A graph search must avoid revisiting nodes to prevent what?|Cleaner names|Cycles causing repeated work|Smaller variables|Sorted edges|2|Boss pressure rises when traversal repeats work."
"For shortest path with nonnegative edge weights, which algorithm fits best?|Dijkstra|Quicksort|Binary search|Reservoir sampling|1|Weighted paths need priority by current distance."
"What is the main tradeoff of memoization?|Less memory for more CPU|More memory for less repeated work|No storage and no speed|Only works on arrays|2|Saved answers cost memory."
"If an O(n) pass is enough, what should you avoid?|A single loop|Extra O(n log n) sorting|Reading input|Comparing values|2|Sorting can dominate the runtime."
)

datastructures_q=(
"Which structure is LIFO?|Queue|Stack|Hash map|Tree|2|hint|The last item added leaves first."
"Which structure is FIFO?|Stack|Heap|Queue|Graph|3|shield|The first item added leaves first."
"What does a hash map optimize for on average?|Lookup by key|Sorted traversal|Recursive calls|Matrix rotation|1|double|Keys map directly to buckets."
"Which structure is natural for hierarchical data?|Array|Tree|Queue|Stack|2|retry|Parents and children form hierarchy."
"A heap is commonly used to implement what?|Priority queue|Linked list|Hash collision chain|Adjacency matrix|1|key|The top item has strongest priority."
)

datastructures_b=(
"A linked list makes which operation cheap when the node is known?|Insert near that node|Binary search|Random access|Sorting by default|1|Pointers can reconnect nearby nodes."
"A graph adjacency list is usually better when edges are what?|Sparse|All possible|Sorted only|Missing weights|1|Sparse graphs avoid huge empty tables."
"Which issue can hurt hash map performance?|Too many collisions|Too many base cases|Balanced tree height|FIFO order|1|Collisions put many keys in the same area."
"Which traversal visits tree levels from top to bottom?|DFS|BFS|Quicksort|Hashing|2|A queue drives level order."
)

generative_ai_q=(
"What improves retrieval-augmented answers most?|Relevant context|Longer random text|Hidden prompts only|More colors|1|hint|The model can only ground on useful context."
"A tool-using agent should do what before a risky action?|Explain intent|Hide the command|Skip validation|Ignore output|1|shield|The user should understand the action."
"What is hallucination?|A true citation|Confident unsupported output|A token counter|A file format|2|double|The answer sounds certain but lacks grounding."
"What should an eval measure?|Only style|Task success against criteria|Screen brightness|Prompt length alone|2|retry|Evaluation needs a target behavior."
"What limits how much context a model can use at once?|Context window|File name|Terminal width|Branch name|1|key|The model has a bounded input budget."
)

generative_ai_b=(
"A strong system instruction should be what?|Clear and scoped|Contradictory|Hidden in output|Unrelated to task|1|Instruction hierarchy works best when clear."
"When retrieval returns weak matches, what should the agent do?|State uncertainty|Invent citations|Ignore the user|Delete files|1|Weak evidence should lower confidence."
"Which pattern helps multi-step agent work?|Plan, act, observe|Guess, hide, finish|Repeat the same command|Avoid feedback|1|Looping through observations improves control."
"Why keep prompts concise?|To reduce useful signal|To improve signal density|To hide requirements|To break tools|2|Concise context reduces noise."
)

machine_learning_q=(
"What is overfitting?|Training too little|Memorizing training data too closely|Using validation data|Cleaning features|2|hint|The model performs poorly outside training data."
"Which split estimates generalization during development?|Training|Validation|Source|Binary|2|shield|Validation checks behavior away from training."
"Precision focuses on what?|Correct positive predictions|All real positives found|File size|Learning rate only|1|double|Precision asks how many predicted positives were right."
"Recall focuses on what?|All real positives found|Only true negatives|Code formatting|Batch names|1|retry|Recall asks how many actual positives were found."
"Which metric fits imbalanced classification better than accuracy alone?|F1|Line count|CPU brand|Disk path|1|key|F1 balances precision and recall."
)

machine_learning_b=(
"A feature should usually be derived from what?|Information available at prediction time|Future labels|Random secrets|Test answers|1|Future data leaks into training."
"Why use a test set once at the end?|To tune forever|To estimate final generalization|To store logs|To rename columns|2|The final set should stay untouched."
"What does regularization usually reduce?|Overfitting|Input rows|Labels|All metrics|1|It constrains the model."
"A confusion matrix shows what?|Prediction counts by class|Only training loss|Disk use|Prompt tokens|1|It breaks outcomes into classes."
)

sre_devops_q=(
"What is an SLO?|A reliability target|A shell alias|A container image|A log color|1|hint|It sets expected service behavior."
"What should observability help answer?|What changed and why|Only file names|Only team titles|Nothing actionable|1|shield|Signals should guide diagnosis."
"What does CrashLoopBackOff usually mean?|A pod repeatedly starts and fails|A sorted queue|A healthy rollout|A completed batch|1|double|The process exits after starting."
"Why keep deployments small?|To reduce blast radius|To hide changes|To skip tests|To increase outage scope|1|retry|Small changes are easier to reason about."
"What is a runbook for?|Repeatable operational response|Random brainstorming|Replacing monitoring|Deleting history|1|key|It guides response under pressure."
)

sre_devops_b=(
"During an incident, what comes first?|Stabilize user impact|Rewrite everything|Assign blame|Ignore alerts|1|Protect the service first."
"Which signal tracks request failure rate?|Errors|Saturation only|Disk names|Branch count|1|Failures are error signals."
"What should a rollback do?|Return to a known good version|Add untested scope|Remove monitoring|Hide symptoms|1|Known good beats uncertain new."
"Why use health checks?|Detect service readiness and liveness|Style shell output|Sort logs|Name pods|1|Health checks feed automation."
)

management_q=(
"Good prioritization starts with what?|Clear goals and tradeoffs|More meetings only|Hidden criteria|Random order|1|hint|Priority needs a decision frame."
"Effective feedback should be what?|Specific and timely|Vague and delayed|Only public|Always avoided|1|shield|Useful feedback names behavior and timing."
"Delegation should include what?|Desired outcome and constraints|Only task title|No ownership|No context|1|double|People need the target and boundaries."
"Stakeholder alignment reduces what?|Surprise and rework|Learning|Team trust|Clear scope|1|retry|Alignment catches mismatch early."
"A healthy plan should include what?|Risks and dependencies|Only dates|No owners|Hidden assumptions|1|key|Plans need the parts that can break."
)

management_b=(
"When priorities conflict, what helps most?|Explicit tradeoff decision|More hidden work|Ignoring one group|Changing nothing|1|Tradeoffs need visibility."
"In conflict resolution, what should a manager seek first?|Shared facts and interests|A quick winner|Silence|Longer status reports|1|Facts and interests reduce heat."
"What is a useful one-on-one pattern?|Listen, clarify, follow through|Talk only about status|Cancel often|Avoid hard topics|1|Trust depends on follow-through."
"What should hiring evaluate?|Role needs and evidence|Vibes only|Speed only|One opinion|1|Evidence anchors the decision."
)

get_entry() {
  array_name="$1"
  index="$2"
  eval "printf '%s' \"\${${array_name}[$index]}\""
}

run_question() {
  entry="$1"
  boss="$2"
  IFS='|' read -r prompt opt1 opt2 opt3 opt4 correct reward hint <<< "$entry"
  tried=false
  while true; do
    card "Quest $((quest_index + 1))" "$(stats_line)" "$prompt" "1. $opt1" "2. $opt2" "3. $opt3" "4. $opt4" "Type h for a hint. Inventory: $(inventory_line)"
    next_input "Your answer: "
    raw="$input_value"
    choice=$(choice_number "$raw")
    if [ "$choice" = "h" ]; then
      if [ "$hint_tokens" -gt 0 ]; then
        hint_tokens=$((hint_tokens - 1))
        card "Hint" "$hint"
      else
        card "Hint" "No Hint Tokens left."
      fi
      continue
    fi
    if [ -z "$choice" ]; then
      card "Invalid Move" "Choose 1, 2, 3, 4, or h."
      continue
    fi
    if [ "$choice" = "$correct" ]; then
      streak=$((streak + 1))
      if [ "$boss" = true ]; then
        boss_health=$((boss_health - 25))
        score=$((score + 150 + streak * 25))
        gain_xp 40
        card "Direct Hit" "Correct. Boss HP $(bar "$boss_health" 100 24)" "$(stats_line)"
      else
        award_item "$reward"
        score=$((score + 100 + streak * 25))
        gain_xp 25
        card "Quest Cleared" "Correct. Reward gained: $awarded_item." "$(stats_line)" "$(inventory_line)"
      fi
      return 0
    fi
    if [ "$tried" = false ] && [ "$retry_charm" -gt 0 ]; then
      retry_charm=$((retry_charm - 1))
      tried=true
      card "Retry Charm" "Wrong answer blocked. Try once more."
      continue
    fi
    streak=0
    if [ "$shields" -gt 0 ]; then
      shields=$((shields - 1))
      card "Shield Block" "Wrong answer. A Shield prevented damage." "$(inventory_line)"
    else
      if [ "$boss" = true ]; then
        health=$((health - 20))
      else
        health=$((health - 10))
      fi
      if [ "$health" -lt 0 ]; then
        health=0
      fi
      card "Hit Taken" "Wrong answer. Correct choice was $correct." "$(stats_line)"
    fi
    return 1
  done
}

command_title() {
  case "$1" in
    algorithms) printf 'Sort a small number stream' ;;
    datastructures) printf 'Trace stack operations' ;;
    generative-ai) printf 'Count agent workflow steps' ;;
    machine-learning) printf 'Count dataset splits' ;;
    sre-devops) printf 'Inspect the local system name' ;;
    management) printf 'Stamp the planning day' ;;
  esac
}

command_text() {
  case "$1" in
    algorithms) printf '%s' "printf '7\\n3\\n9\\n1\\n' | sort -n" ;;
    datastructures) printf '%s' "printf 'push alpha\\npush beta\\npop beta\\n'" ;;
    generative-ai) printf '%s' "printf 'prompt\\nretrieve\\ntool\\nevaluate\\n' | wc -l" ;;
    machine-learning) printf '%s' "printf 'train\\nvalidation\\ntest\\n' | wc -l" ;;
    sre-devops) printf '%s' "uname -s" ;;
    management) printf '%s' "date '+%A'" ;;
  esac
}

run_theme_command() {
  case "$1" in
    algorithms) printf '7\n3\n9\n1\n' | sort -n ;;
    datastructures) printf 'push alpha\npush beta\npop beta\n' ;;
    generative-ai) printf 'prompt\nretrieve\ntool\nevaluate\n' | wc -l ;;
    machine-learning) printf 'train\nvalidation\ntest\n' | wc -l ;;
    sre-devops) uname -s ;;
    management) date '+%A' ;;
  esac
}

command_quest() {
  selected="$1"
  card "Command Quest" "$(command_title "$selected")" "Command: $(command_text "$selected")" "This command is fixed and read-only for this game."
  next_input "Run command quest? y/N: "
  raw="$input_value"
  answer=$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')
  case "$answer" in
    y|yes)
      output=$(run_theme_command "$selected" 2>&1)
      output=$(printf '%s' "$output" | tr '\n' ' ')
      double_xp=$((double_xp + 1))
      score=$((score + 75))
      card "Command Cleared" "Output: $output" "Reward gained: Double XP." "$(stats_line)"
      ;;
    *)
      card "Command Skipped" "No penalty. The quest path continues."
      ;;
  esac
}

history_count() {
  selected="$1"
  file="$state_dir/$selected.jsonl"
  if [ -f "$file" ]; then
    wc -l < "$file" | tr -d ' '
  else
    printf '0'
  fi
}

persist_history() {
  selected="$1"
  mkdir -p "$state_dir" 2>/dev/null || return
  run_id="$(date -u '+%Y%m%dT%H%M%SZ')-$$"
  completed_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  file="$state_dir/$selected.jsonl"
  printf '{"run_id":"%s","theme":"%s","level":%s,"xp":%s,"score":%s,"health":%s,"boss_defeated":%s,"completed_at":"%s"}\n' "$run_id" "$selected" "$level" "$xp" "$score" "$health" "$boss_defeated" "$completed_at" >> "$file"
}

intro() {
  selected="$1"
  title=$(theme_title "$selected")
  past=$(history_count "$selected")
  paint '1;36' "$skill_name"
  printf '\n'
  sleep_step
  card "$title Realm" "New game started. Past completed runs for this theme: $past." "Win quests, collect items, survive the boss fight." "$(stats_line)"
}

play_theme() {
  selected="$1"
  questions="${selected//-/_}_q"
  bosses="${selected//-/_}_b"
  if [ "$selected" = "datastructures" ]; then
    questions="datastructures_q"
    bosses="datastructures_b"
  fi
  if [ "$selected" = "sre-devops" ]; then
    questions="sre_devops_q"
    bosses="sre_devops_b"
  fi
  q_total=5
  quest_index=0
  while [ "$quest_index" -lt "$q_total" ]; do
    entry=$(get_entry "$questions" "$quest_index")
    run_question "$entry" false
    quest_index=$((quest_index + 1))
    render_status
    if [ "$quest_index" -eq 2 ]; then
      command_quest "$selected"
    fi
    if [ "$health" -le 0 ]; then
      return
    fi
  done
  if [ "$boss_key" -gt 0 ]; then
    card "Boss Gate" "Boss Key activates. Final fight opens."
  else
    card "Boss Gate" "Five quests cleared. Final fight opens."
  fi
  b_index=0
  while [ "$b_index" -lt 4 ] && [ "$boss_health" -gt 0 ] && [ "$health" -gt 0 ]; do
    entry=$(get_entry "$bosses" "$b_index")
    run_question "$entry" true
    b_index=$((b_index + 1))
  done
  if [ "$boss_health" -le 0 ]; then
    boss_defeated=true
    score=$((score + 500))
    gain_xp 100
  fi
}

finish() {
  selected="$1"
  persist_history "$selected"
  if [ "$boss_defeated" = true ]; then
    result="Victory. Boss defeated."
  elif [ "$health" -le 0 ]; then
    result="Defeat. Health reached zero."
  else
    result="Run complete."
  fi
  card "Final Score" "$result" "$(stats_line)" "$(inventory_line)" "History saved by theme in $state_dir."
}

main() {
  selected=$(choose_theme)
  intro "$selected"
  play_theme "$selected"
  finish "$selected"
}

main "$@"
