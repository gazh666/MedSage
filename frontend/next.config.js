/** @type {import('next').NextConfig} */
const configuredApiOrigin = (() => {
  try {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    return apiUrl ? new URL(apiUrl).origin : null;
  } catch {
    return null;
  }
})();

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  compress: true,
  output: 'standalone',
  
  // Environment variables for client-side access
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
  
  // Modern Image Optimization (replaces 'domains')
  images: {
    formats: ['image/avif', 'image/webp'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**', // Recommended: Replace with your specific S3/Cloudinary bucket in production
      },
    ],
  },
  
  // Production Security Headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on'
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload'
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block'
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin'
          },
          {
            key: 'Content-Security-Policy',
            // Refined policy to allow API calls and Google AI Studio assets
            // We use the base domain in connect-src to allow all subpaths naturally
            value: `default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self' https://*.onrender.com http://*:8000${configuredApiOrigin ? ` ${configuredApiOrigin}` : ''};`
          }
        ],
      },
    ]
  },
  
  // Configuration for large file uploads (X-Rays/PDFs)
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },
}

module.exports = nextConfig