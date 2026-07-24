<?php
/**
 * @return array{0: string|false, 1: int, 2: string} [$body, $httpStatus, $error]
 */
function winicari_http_request(
    string $url,
    array $headers = [],
    string $method = 'GET',
    ?string $body = null,
    float $timeoutSeconds = 90.0,
    bool $verifySsl = true
): array {
    $http_opts = [
        'method' => $method,
        'header' => implode("\r\n", $headers),
        'timeout' => $timeoutSeconds,
        'ignore_errors' => true, 
    ];
    if ($body !== null) {
        $http_opts['content'] = $body;
    }
    $context = stream_context_create([
        'http' => $http_opts,
        'ssl' => [
            'verify_peer' => $verifySsl,
            'verify_peer_name' => $verifySsl,
        ],
    ]);

    $result = @file_get_contents($url, false, $context);

    $status = 0;
    foreach ($http_response_header ?? [] as $h) {
        if (preg_match('#^HTTP/\S+\s+(\d{3})#', $h, $m)) {
            $status = (int)$m[1]; // last one wins (redirects leave earlier ones in the array)
        }
    }

    if ($result === false) {
        $err = error_get_last();
        return [false, $status, $err['message'] ?? 'stream request failed'];
    }
    return [$result, $status, ''];
}
