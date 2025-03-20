use std::cmp::PartialEq;
use std::io;
use std::io::Write;
use std::iter::Peekable;
use std::str::CharIndices;
use std::time::Instant;

fn main() {
    println!("Enter your mathematical expression:");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input)
        .expect("Failed to read line");

    // Track time
    let time_start = Instant::now();
    let result = calculate(input.trim());
    let duration = time_start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Err(message) => println!("Error: {}", message),
        Ok(result) => println!("{}", result)
    }

    println!("Done in {:.5}ms", duration);
}

/// Calculator abstract syntax token. Can be number, operator or parentheses bracket
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Number(f64),

    /* Operators */
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Pow,
    Factorial,

    /* Parentheses */
    LeftParen,
    RightParen,
}

/// Error that can appear while calculating some expression
#[derive(Debug, PartialEq)]
pub enum CalculateError<'a> {
    InvalidCharacter(char),
    InvalidNumber(&'a str),
    MismatchedParenthesis,
    DivideByZero,
    ModuloByZero,
    InvalidExpression,
    MalformedTokens,
    ConsumingError
}

impl std::fmt::Display for CalculateError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CalculateError::InvalidCharacter(c) => write!(f, "Invalid character '{}'", c),
            CalculateError::InvalidNumber(s) => write!(f, "Invalid number '{}'", s),
            CalculateError::MismatchedParenthesis => write!(f, "Mismatched parenthesis"),
            CalculateError::DivideByZero => write!(f, "Divide by zero"),
            CalculateError::ModuloByZero => write!(f, "Modulo by zero"),
            CalculateError::InvalidExpression => write!(f, "Invalid expression"),
            CalculateError::MalformedTokens => write!(f, "Malformed tokens"),
            CalculateError::ConsumingError => write!(f, "Consuming error"),
        }
    }
}

/// Tokenize your mathematical expression to vector of token enums
/// ## Example
///
/// ```rust
/// let expression: &str = "1 - 89 * (23 + 2)";
/// let tokens: Vec<Token> = tokenize_expression(expression);
/// println!("Tokens: {}", tokens);
/// ```
/// We should get:
///
/// `Token::Number(1)`, `Token::Minus`, `Token::Number(89)`, `Token::Multiply`,
/// `Token::LeftParen`, `Token::Number(23)`, `Token::Plus`, `Token::Number(2)`,
/// `Token::RightParen`
fn tokenize_expression(expression: &str) -> Result<Vec<Token>, CalculateError> {
    // Capacity of len/2 should be enough for most cases
    let mut tokens: Vec<Token> = Vec::with_capacity(expression.len());
    let mut chars = expression.char_indices().peekable();

    while let Some((s, c)) = chars.next() {
        // Skip whitespaces
        if c.is_whitespace() {
            continue;
        }

        // Parsing positive number
        if c.is_ascii_digit() {
            let number = parse_number(expression, &mut chars, s)?;
            tokens.push(Token::Number(number));
            continue;
        }

        match c {
            '+' => tokens.push(Token::Plus),
            '*' => tokens.push(Token::Multiply),
            '/' => tokens.push(Token::Divide),
            '%' => tokens.push(Token::Modulo),
            '(' => tokens.push(Token::LeftParen),
            ')' => tokens.push(Token::RightParen),
            '^' => tokens.push(Token::Pow),
            '-' => {
                if let Some((_, next_char)) = chars.peek() {
                    // Checking if this - symbol is actually part of some negative number
                    if next_char.is_ascii_digit() {
                        if let Some(last_token) = tokens.last() {
                            // Negative numbers allowed only in parentheses
                            if *last_token == Token::LeftParen {
                                let number = parse_number(expression, &mut chars, s)?;
                                tokens.push(Token::Number(number));
                                continue;
                            }
                        }
                        return Err(CalculateError::InvalidCharacter(c));
                    }
                }
                tokens.push(Token::Minus);
            }
            _ => return Err(CalculateError::InvalidCharacter(c)),
        }
    }
    Ok(tokens)
}

/// Parsing number as f64
fn parse_number<'a>(
    expression: &'a str,
    chars: &mut Peekable<CharIndices>,
    s: usize
) -> Result<f64, CalculateError<'a>> {
    let mut token_end = s + 1;
    // Here we're reading number from left to right to find its start and end
    while let Some((pos, c)) = chars.peek() {
        if c.is_ascii_digit() || *c == '.' {
            token_end = *pos + 1;
            chars.next();
        } else {
            break;
        }
    }

    let number_slice = &expression[s..token_end];
    // Parsing number to f64
    let number = number_slice.parse::<f64>().map_err(|_| {
        CalculateError::InvalidNumber(number_slice)
    })?;
    Ok(number)
}

/// Converts infix notation tokens to postfix notation (Reverse Polish Notation).
/// Returns a vector where numbers are located on the left side in order to calculate, and operators located on the right side in order of calculation rules.
/// Requires ownership of tokens for zero-copy transformation
/// ## Example
/// ```rust
/// let tokens = vec![
///     Token::Number(4),
///     Token::Plus,
///     Token::LeftParen,
///     Token::Number(12),
///     Token::Minus,
///     Token::Number(7),
///     Token::RightParen
/// ];
/// let tokens_in_order_to_calculate = shunting_yard(tokens)?;
/// println!("{}", tokens_in_order_to_calculate);
/// ```
pub(crate) fn shunting_yard<'a>(mut tokens: Vec<Token>) -> Result<Vec<Token>, CalculateError<'a>> {
    let mut output: Vec<Token> = Vec::with_capacity(tokens.len());
    // Buffer for correct order of operators
    let mut operator_stack = Vec::new();
    let mut index = 0;

    while index < tokens.len() {
        // Raw pointer swapping
        let token = unsafe { std::ptr::read(tokens.as_ptr().add(index)) };
        index += 1;

        match token {
            Token::Number(_) => output.push(token),
            // Pushing opening parentheses bracket only to operator stack since it will be used only as mark of parentheses start
            Token::LeftParen => operator_stack.push(token),
            // Right parentheses brackets represent end of parentheses, so we add all its tokens to stack
            Token::RightParen => {
                let mut opening_bracket_found = false;
                // Adding all tokens while we don't find opening bracket
                while let Some(token) = operator_stack.pop() {
                    if matches!(token, Token::LeftParen) {
                        opening_bracket_found = true;
                        break;
                    }
                    output.push(token);
                }
                if !opening_bracket_found {
                    return Err(CalculateError::MismatchedParenthesis);
                }
            },
            _ => {
                let current_precedence = precedence(&token);
                while let Some(last_op) = operator_stack.last() {
                    // Compare using precedence values
                    if matches!(last_op, Token::LeftParen)
                        || precedence(last_op) < current_precedence
                    {
                        break;
                    }
                    output.push(operator_stack.pop().unwrap());
                }
                operator_stack.push(token);
            }
        }
    }

    // Reuse operator stack directly
    output.extend(operator_stack.into_iter().rev().filter(|op| {
        !matches!(op, Token::LeftParen | Token::RightParen)
    }));

    // Reclaim memory from original tokens vector
    tokens.clear();
    tokens.shrink_to_fit();

    Ok(output)
}

/// Get precedence of operators. Plus and minus will return 1. Multiply, divide and module will return 2. Others will return 0
fn precedence(token: &Token) -> u8 {
    match token {
        Token::Plus | Token::Minus => 1,
        Token::Multiply | Token::Divide | Token::Modulo => 2,
        Token::Pow => 3,
        _ => 0,
    }
}

pub(crate) fn evaluate_postfix<'a>(tokens: &[Token]) -> Result<f64, CalculateError<'a>> {
    let mut stack: Vec<f64> = Vec::with_capacity((tokens.len() + 1) / 2);

    for token in tokens {
        match token {
            Token::Number(n) => stack.push(*n),
            Token::Plus => {
                let (left, right) = pop_two(&mut stack)?;
                stack.push(left + right);
            },
            Token::Minus => {
                let (left, right) = pop_two(&mut stack)?;
                stack.push(left - right);
            }
            Token::Multiply => {
                let (left, right) = pop_two(&mut stack)?;
                stack.push(left * right);
            }
            Token::Divide => {
                let (left, right) = pop_two(&mut stack)?;
                if right == 0.0 {
                    return Err(CalculateError::DivideByZero);
                }
                stack.push(left / right);
            }
            Token::Modulo => {
                let (left, right) = pop_two(&mut stack)?;
                if right == 0.0 {
                    return Err(CalculateError::ModuloByZero);
                }
                stack.push(left % right);
            },
            Token::Pow => {
                let (left, right) = pop_two(&mut stack)?;
                stack.push(left.powf(right));
            }
            _ => Err(CalculateError::InvalidExpression)?,
        }
    }

    // Consume the final value from stack
    stack.pop().ok_or(CalculateError::ConsumingError)
}

/// Helper function to safely pop two operands
fn pop_two<'a>(stack: &mut Vec<f64>) -> Result<(f64, f64), CalculateError<'a>> {
    let right = stack.pop().ok_or(CalculateError::MalformedTokens)?;
    let left = stack.pop().ok_or(CalculateError::MalformedTokens)?;
    Ok((left, right))
}

pub fn calculate(expression: &str) -> Result<f64, CalculateError> {
    let tokens = tokenize_expression(expression)?;
    let tokens_to_calculate = shunting_yard(tokens)?;
    evaluate_postfix(&tokens_to_calculate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow_test() {
        assert_eq!(calculate("2^2").unwrap(), 2.0f64.powf(2.0));
    }

    #[test]
    fn multiply_test() {
        assert_eq!(calculate("2*2").unwrap(), 4.0f64);
    }

    #[test]
    fn div_test() {
        assert_eq!(calculate("2/2").unwrap(), 1.0f64);
    }

    #[test]
    fn modulo_test() {
        assert_eq!(calculate("2%2").unwrap(), 0.0f64);
    }

    #[test]
    fn tokenizer_test() {
        assert_eq!(
            tokenize_expression("2 + 2 * (67.1 - 1)").unwrap(),
            vec![
                Token::Number(2.0),
                Token::Plus,
                Token::Number(2.0),
                Token::Multiply,
                Token::LeftParen,
                Token::Number(67.1),
                Token::Minus,
                Token::Number(1.0),
                Token::RightParen,
            ]
        );
    }

    #[test]
    fn evaluate_postfix_test() {
        assert_eq!(
            shunting_yard(vec![
                Token::Number(2.0),
                Token::Plus,
                Token::Number(4.0),
            ]).unwrap(),
            vec![
                Token::Number(2.0),
                Token::Number(4.0),
                Token::Plus,
            ]
        );
    }

    #[test]
    fn complex_calculation_test() {
        assert_eq!(
            calculate("(( (3^2 + 4) * 2 )^4 ) / (5 - 2^3) + 10 * (4 - 1)^2"),
            Ok(-152235.33333333334f64)
        )
    }
}